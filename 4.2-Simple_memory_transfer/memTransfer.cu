#include <cuda_runtime.h>
#include <stdio.h>

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

int main(int argc, char **argv)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev,
           deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    float *h_a = (float *)malloc(nbytes);
    float *d_a;
    CHECK(cudaMalloc((float **)&d_a, nbytes));

    for (unsigned int i = 0; i < isize; i++)
        h_a[i] = 0.5f;

    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    free(h_a);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}