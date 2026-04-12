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
// *ip is the pointer to the array, and size is number of elements in that array. Size is essential to prevent overflow or not filling up whole array
void initialData(float *ip, int size) {
    // This initializes rand() function with a seed, change the seed to time(NULL) to use current time as seed which makes it whole lot more randomized on executions
    srand(11);
    for (int i = 0; i < size; i++)
    {
        // First rand() generates an integer which is clipped to 8bits by bitwise AND, resulting in integer between 0 - 255, this is then converted into float. This is again scaled down by the factor of 10. Example: some random number -> (after bitwise AND) 255 -> 255.00 -> 25.50
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Main Function
int main() {
    // Number of elements in the array
    int nElem = 65536;
    // Calculating the size of that array if the data type of the array is float
    size_t nBytes = nElem * sizeof(float);

    // Initializing the array variables
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);

    // We'll use this variable to store the result fetched from GPU to host
    float *h_D = (float *)malloc(nBytes);

    // Declare variables for cudaMalloc
    float *d_A, *d_B, *d_C;

    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // Array that has been populated with the randomly generated floats is copied over to GPU
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // CPU Timer
    double cpuStart = cpuTimer();
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    double cpuEnd = cpuTimer();
    printf("CPU finished Sum Array Operation  |  Time: %.3f ms\n", cpuEnd - cpuStart);


    // These values multiplied are threads, which must be equal to or greater than nElem to get correct answer
    // Block size should preferably be the multiple of 32 (128, 256, or 512). You'll learn about this warp size (32) in subsequent programs
    double gpuStart = cpuTimer();
    sumArraysOnGPU<<<512, 128>>>(d_A, d_B, d_C, nElem);
    double timeTaken = cpuTimer() - gpuStart;

    printf("GPU finished Sum Array Operation (CPU Timer)  |  Time: %.3f ms\n", timeTaken);

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
