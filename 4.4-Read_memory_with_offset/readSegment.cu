#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

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

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N, int offset)
{
    for (int idx = offset, k = 0; idx < N; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int N, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < N)
    {
        C[i] = A[k] + B[k];
    }
}

__global__ void readOffset(float *A, float *B, float *C, const int N, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < N)
    {
        C[i] = A[k] + B[k];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
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
    printf("%s starting reduction at device %d: %s ", argv[0], dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 20;
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    int blocksize = 512;
    int offset = 0;
    if (argc > 1)
        offset = atoi(argv[1]);
    if (argc > 2)
        blocksize = atoi(argv[2]);

    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}