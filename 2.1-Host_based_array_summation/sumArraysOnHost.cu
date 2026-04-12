// Using the C++ form headers
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <sys/time.h>

// Returns current time in microseconds
double cpuTimer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0; // returns milliseconds
}

// CUDA Kernel
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// The Name of the function says it all
void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    // The whole operation here is sequential
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// This function is used to populate an array with random numbers.
void initialData(float *ip, int size) {
    srand(11);
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Main Function
int main() {
    int nElem = 65536;
    size_t nBytes = nElem * sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);
    float *h_D = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // CPU Timer
    double cpuStart = cpuTimer();
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    double cpuEnd = cpuTimer();
    printf("CPU finished Sum Array Operation  |  Time: %.3f ms\n", cpuEnd - cpuStart);

    // GPU Timer using CUDA Events
    cudaEvent_t gpuStart, gpuEnd;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuEnd);

    cudaEventRecord(gpuStart);
    sumArraysOnGPU<<<512, 128>>>(d_A, d_B, d_C, nElem);
    cudaEventRecord(gpuEnd);

    // Blocks CPU until the GPU event is recorded
    cudaEventSynchronize(gpuEnd);

    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, gpuStart, gpuEnd);
    printf("GPU finished Sum Array Operation  |  Time: %.3f ms\n", gpuMs);

    // Cleanup events
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);

    // Copy back the results from the GPU to host
    cudaMemcpy(h_D, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Check if the results are same or different
    for (int i = 0; i < nElem; i++) {
        if (h_C[i] != h_D[i]) {
            printf("Error\n");
            break;
        }
    }

    // Free up Allocations
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return (0);
}
