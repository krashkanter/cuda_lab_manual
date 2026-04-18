#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#define LEN 1 << 20

struct innerStruct {
    float x;
    float y;
};

inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialInnerStruct(innerStruct *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i].x = (float)(rand() & 0xFF) / 10.0f;
        ip[i].y = (float)(rand() & 0xFF) / 10.0f;
    }
}

void testInnerStructHost(innerStruct *A, innerStruct *C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        C[idx].x = A[idx].x + 10.0f;
        C[idx].y = A[idx].y + 20.0f;
    }
}

__global__ void warmup(innerStruct *A, innerStruct *C, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i].x = A[i].x + 10.0f;
        C[i].y = A[i].y + 20.0f;
    }
}

__global__ void testInnerStruct(innerStruct *A, innerStruct *C, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i].x = A[i].x + 10.0f;
        C[i].y = A[i].y + 20.0f;
    }
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i].x - gpuRef[i].x) > epsilon || fabs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("Arrays do not match! at %d\n", i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

int main() {
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *)malloc(nBytes);

    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    innerStruct *d_A, *d_C;
    cudaMalloc((innerStruct **)&d_A, nBytes);
    cudaMalloc((innerStruct **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksize = 128;


    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    iStart = seconds();
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n", grid.x, block.x, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_C);

    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
