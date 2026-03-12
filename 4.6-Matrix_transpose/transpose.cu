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

void transposeHost(float *out, float *in, int nx, int ny)
{
    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void warmup(float *out, float *in, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyRow(float *out, float *in, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float *out, float *in, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N, int matchCheck)
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
    printf("%s starting transpose at device %d: %s \n", argv[0], dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 11;
    int ny = 1 << 11;

    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1)
        iKernel = atoi(argv[1]);
    if (argc > 2)
        blockx = atoi(argv[2]);
    if (argc > 3)
        blocky = atoi(argv[3]);
    if (argc > 4)
        nx = atoi(argv[4]);
    if (argc > 5)
        ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nx * ny);

    transposeHost(hostRef, h_A, nx, ny);

    float *d_A, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    double iStart = seconds();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup elapsed %f sec\n", iElaps);

    void (*kernel)(float *, float *, int, int);
    char *kernelName = "";

    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow";
        break;
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol";
        break;
    }

    iStart = seconds();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("%s elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> effective bandwidth %f GB\n",
           kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);

    if (iKernel > 1)
    {
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}