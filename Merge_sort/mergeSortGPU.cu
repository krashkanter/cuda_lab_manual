#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(int *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = rand() % 10000;
    }
}

void cpu_bottomUpMerge(int *source, int *dest, int start, int middle, int end)
{
    int i = start;
    int j = middle;
    int k = start;

    while (i < middle && j < end)
    {
        if (source[i] <= source[j])
        {
            dest[k++] = source[i++];
        }
        else
        {
            dest[k++] = source[j++];
        }
    }

    while (i < middle)
    {
        dest[k++] = source[i++];
    }

    while (j < end)
    {
        dest[k++] = source[j++];
    }
}

void cpu_mergeSort(int *arr, int size)
{
    int *temp = (int *)malloc(size * sizeof(int));
    int *A = arr;
    int *B = temp;

    for (int width = 1; width < size; width *= 2)
    {
        for (int i = 0; i < size; i += 2 * width)
        {
            int start = i;
            int middle = start + width;
            int end = start + 2 * width;

            if (middle > size)
                middle = size;
            if (end > size)
                end = size;

            cpu_bottomUpMerge(A, B, start, middle, end);
        }
        int *tmp = A;
        A = B;
        B = tmp;
    }

    if (A != arr)
    {
        memcpy(arr, A, size * sizeof(int));
    }
    free(temp);
}

__global__ void gpu_bottomUpMerge(int *source, int *dest, int width, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * 2 * width;

    if (start >= size)
        return;

    int middle = start + width;
    int end = start + 2 * width;

    if (middle > size)
        middle = size;
    if (end > size)
        end = size;

    int i = start;
    int j = middle;
    int k = start;

    while (i < middle && j < end)
    {
        if (source[i] <= source[j])
        {
            dest[k++] = source[i++];
        }
        else
        {
            dest[k++] = source[j++];
        }
    }

    while (i < middle)
    {
        dest[k++] = source[i++];
    }

    while (j < end)
    {
        dest[k++] = source[j++];
    }
}

void checkResult(int *hostRef, int *gpuRef, const int N)
{
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            match = 0;
            printf("Arrays do not match! at %d: host %d gpu %d\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int size = 1 << 20;
    printf("Data size %d\n", size);

    size_t nBytes = size * sizeof(int);
    int *h_A = (int *)malloc(nBytes);
    int *hostRef = (int *)malloc(nBytes);
    int *gpuRef = (int *)malloc(nBytes);

    initialData(h_A, size);
    memcpy(hostRef, h_A, nBytes);
    memcpy(gpuRef, h_A, nBytes);

    double iStart, iElaps;

    iStart = seconds();
    cpu_mergeSort(hostRef, size);
    iElaps = seconds() - iStart;
    printf("CPU Merge Sort elapsed %f sec\n", iElaps);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    int blocksize = 256;
    if (argc > 1)
        blocksize = atoi(argv[1]);

    iStart = seconds();

    for (int width = 1; width < size; width *= 2)
    {
        int numThreads = (size + (2 * width) - 1) / (2 * width);
        dim3 block(blocksize, 1);
        dim3 grid((numThreads + block.x - 1) / block.x, 1);

        gpu_bottomUpMerge<<<grid, block>>>(d_A, d_B, width, size);
        CHECK(cudaDeviceSynchronize());

        int *d_temp = d_A;
        d_A = d_B;
        d_B = d_temp;
    }

    if (d_A != d_B)
    {
        CHECK(cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost));
    }
    else
    {
        CHECK(cudaMemcpy(gpuRef, d_B, nBytes, cudaMemcpyDeviceToHost));
    }

    iElaps = seconds() - iStart;
    printf("GPU Merge Sort elapsed %f sec\n", iElaps);

    checkResult(hostRef, gpuRef, size);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}