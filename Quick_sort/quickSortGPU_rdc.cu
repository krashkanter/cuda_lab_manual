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

void cpu_quickSort(int *arr, int left, int right)
{
    int i = left, j = right;
    int pivot = arr[left + (right - left) / 2];

    while (i <= j)
    {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }

    if (left < j)
        cpu_quickSort(arr, left, j);
    if (i < right)
        cpu_quickSort(arr, i, right);
}

__device__ void seq_quickSort(int *arr, int left, int right)
{
    int stack[128];
    int top = -1;

    stack[++top] = left;
    stack[++top] = right;

    while (top >= 0)
    {
        int r = stack[top--];
        int l = stack[top--];
        int i = l, j = r;
        int pivot = arr[l + (r - l) / 2];

        while (i <= j)
        {
            while (arr[i] < pivot)
                i++;
            while (arr[j] > pivot)
                j--;
            if (i <= j)
            {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
                j--;
            }
        }

        if (l < j)
        {
            stack[++top] = l;
            stack[++top] = j;
        }
        if (i < r)
        {
            stack[++top] = i;
            stack[++top] = r;
        }
    }
}

__global__ void gpu_quickSort(int *arr, int left, int right, int depth)
{
    if (left >= right)
        return;

    if (right - left < 2048 || depth > 24)
    {
        seq_quickSort(arr, left, right);
        return;
    }

    int i = left, j = right;
    int pivot = arr[left + (right - left) / 2];

    while (i <= j)
    {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }

    if (left < j)
        gpu_quickSort<<<1, 1>>>(arr, left, j, depth + 1);
    if (i < right)
        gpu_quickSort<<<1, 1>>>(arr, i, right, depth + 1);
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

    CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768));

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
    cpu_quickSort(hostRef, 0, size - 1);
    iElaps = seconds() - iStart;
    printf("CPU Quick Sort elapsed %f sec\n", iElaps);

    int *d_A;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    iStart = seconds();
    gpu_quickSort<<<1, 1>>>(d_A, 0, size - 1, 0);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("GPU Quick Sort elapsed %f sec\n", iElaps);

    CHECK(cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, size);

    CHECK(cudaFree(d_A));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}