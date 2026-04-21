#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1024

// Kernel for unoptimized matrix multiplication
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
    
    // Allocate host memory
    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    // Allocate device memory
    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void**)&gpu_a, size);
    cudaMalloc((void**)&gpu_b, size);
    cudaMalloc((void**)&gpu_c, size);

    // Copy matrices to device
    cudaMemcpy(gpu_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);

    // Configuration from problem statement: 16x16 blocks, 64x64 grid
    dim3 dimBlock(16, 16);
    dim3 dimGrid(64, 64);

    struct timespec start, stop;
    double time;

    // Launch kernel and measure time
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    
    matmul_global<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize(); // Ensure kernel completes before stopping clock
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
    
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    
    // Copy result back to host
    cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

    // Required output
    printf("Execution Time (Global): %f ns\n", time * 1e9);
    printf("C[451][451] = %d\n", c[451 * N + 451]);

    // Free memory
    free(a); free(b); free(c);
    cudaFree(gpu_a); cudaFree(gpu_b); cudaFree(gpu_c);

    return 0;
}