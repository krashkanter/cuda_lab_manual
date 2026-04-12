#include <stdio.h>

#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

__global__ void helloFromGPU(int n)
{
    printf("Hello World from GPU. %d\n", n);
}

// Main Function
int main()
{
    // Here Thread count is valid, so execution is done
    printf("Executing first batch\n");
    helloFromGPU<<<1, 2>>>(1);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaGetLastError());

    printf("Executing second batch\n");
    // Here an invalid thread count is given, fails silently if Check isnt used
    helloFromGPU<<<10240, 10240>>>(2);

    // Using check to catch the errors
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    cudaDeviceReset();

    return 0;
}

