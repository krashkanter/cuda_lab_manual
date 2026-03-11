#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv)
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnGPU<<<1, 10>>>(d_A, d_B, d_C, nElem);

    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (0);
}