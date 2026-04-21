#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N 1024

__global__ void matmul_global(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);
    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void**)&gpu_a, size);
    cudaMalloc((void**)&gpu_b, size);
    cudaMalloc((void**)&gpu_c, size);
    cudaMemcpy(gpu_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(64, 64);

    struct timespec start, stop;
    double time;

    clock_gettime( CLOCK_REALTIME, &start)
    matmul_global<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();    
    clock_gettime( CLOCK_REALTIME, &stop)
    
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    
    cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

    printf("execution time (global): %f ns\n", time * 1e9);
    printf("C[451][451] = %d\n", c[451 * N + 451]);

    free(a); 
    free(b); 
    free(c);
    cudaFree(gpu_a); 
    cudaFree(gpu_b); 
    cudaFree(gpu_c);

    return 0;
}