#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 1024
#define TILE_WIDTH 32

// Kernel for tiled matrix multiplication using shared memory
__global__ void matmul_shared(int *a, int *b, int *c) {
    // Allocate shared memory for tiles
    __shared__ int subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int subTileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Identify row and col to work on
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    int sum = 0;

    // Loop over all tiles required to compute the C element
    for (int phase = 0; phase < N / TILE_WIDTH; phase++) {
        
        // Load data into shared memory
        subTileA[ty][tx] = a[row * N + (phase * TILE_WIDTH + tx)];
        subTileB[ty][tx] = b[(phase * TILE_WIDTH + ty) * N + col];
        
        // Synchronize to make sure the tile is completely loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += subTileA[ty][k] * subTileB[k][tx];
        }

        // Synchronize to make sure computation is done before loading the next tile
        __syncthreads();
    }

    if (row < N && col < N) {
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

    // Configuration from problem statement: 32x32 blocks, 32x32 grid
    dim3 dimBlock(32, 32);
    dim3 dimGrid(32, 32);

    struct timespec start, stop;
    double time;

    // Launch kernel and measure time
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    
    matmul_shared<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaDeviceSynchronize(); // Ensure kernel completes before stopping clock
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
    
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    
    // Copy result back to host
    cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

    // Required output
    printf("Execution Time (Shared): %f ns\n", time * 1e9);
    printf("C[451][451] = %d\n", c[451 * N + 451]);

    // Free memory
    free(a); free(b); free(c);
    cudaFree(gpu_a); cudaFree(gpu_b); cudaFree(gpu_c);

    return 0;
}