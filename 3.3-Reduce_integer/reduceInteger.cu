#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int recursiveReduce(int *data, int const size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnroll8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at device %d: %s \n", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;
    int size = 1 << 24;
    printf("with array size %d \n", size);

    int blocksize = 512;
    if (argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Warmup elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnroll8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];
    printf("gpu Cmptnroll elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    free(h_idata);
    free(h_odata);
    free(tmp);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    bResult = (gpu_sum == cpu_sum);
    if (!bResult)
        printf("Test failed!\n");
    else
        printf("Test passed!\n");

    return EXIT_SUCCESS;
}
