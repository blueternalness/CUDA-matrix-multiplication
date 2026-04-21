#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1024
#define TILE_WIDTH 32

__global__ void matmul_shared(int *a, int *b, int *c) {
    __shared__ int subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int subTileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int sum = 0;

    for (int phase = 0; phase < N / TILE_WIDTH; phase++) {
        subTileA[ty][tx] = a[row * N + (phase * TILE_WIDTH + tx)];
        subTileB[ty][tx] = b[(phase * TILE_WIDTH + ty) * N + col];
        
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
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
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);
    struct timespec start, stop;
    double time;

    clock_gettime( CLOCK_REALTIME, &start);
    matmul_shared<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize();
    clock_gettime( CLOCK_REALTIME, &stop);
    
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    
    cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

    printf("execution time (shared): %f ns\n", time * 1e9);
    printf("C[451][451] = %d\n", c[451 * N + 451]);

    free(a); 
    free(b); 
    free(c);
    cudaFree(gpu_a); 
    cudaFree(gpu_b); 
    cudaFree(gpu_c);

    return 0;
}