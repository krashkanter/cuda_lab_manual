#include <stdio.h>


// A CUDA kernel which runs once per thread
// Your First CUDA Kernel :)
__global__ void helloFromGPU()
{
    printf("Hello World from GPU\n");
}


// Main Function
int main()
{
    // This line of code is executed from CPU
    printf("Hello World from CPU\n");

    // CUDA kernel is invoked here
    // Grid Size = 1, Block Size = 10 ==> 10 Threads
    helloFromGPU<<<1, 10>>>();

    // 1 Time Hello from CPU and 10 Times Hello from GPU is printed

    // This is essential to clean up the CUDA context (like memory allocations on GPU)
    cudaDeviceReset();

    return 0;
}
