#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N, int offset) {
    for (int idx = offset, k = 0; idx < N; idx++, k++) {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int N, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < N) {
        C[i] = A[k] + B[k];
    }
}

__global__ void readOffset(float *A, float *B, float *C, const int N, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < N) {
        C[i] = A[k] + B[k];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

int main(int argc, char **argv) {
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
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice);

    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem - offset);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
