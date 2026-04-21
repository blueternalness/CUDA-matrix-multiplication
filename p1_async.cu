#include <stdio.h>
#include <stdlib.h>

#define N 1024

// Kernel for matrix multiplication computing a specific chunk of rows (horizontal tile)
__global__ void matmul(int *A, int *B, int *C, int row_offset) {
    // Map x to rows and y to columns to match the requested (64/nStreams) x 64 grid
    int row = row_offset + blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_test(int nStreams, int *h_A, int *h_B, int *h_C, int *d_A, int *d_B, int *d_C) {
    int streamSize = (N / nStreams) * N; // Number of elements per stream
    int streamBytes = streamSize * sizeof(int);

    // Create Streams
    cudaStream_t streams[16];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Create Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // B is needed entirely by all streams, so we copy it once synchronously before the timer
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Configure dimensions
    dim3 dimBlock(16, 16);
    dim3 dimGrid(64 / nStreams, 64);

    cudaEventRecord(start);

    // --- SEQUENTIAL ISSUE LOOP ---
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        int row_offset = i * (N / nStreams);

        // H2D 
        cudaMemcpyAsync(&d_A[offset], &h_A[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        
        // Kernel
        matmul<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A, d_B, d_C, row_offset);
        
        // D2H
        cudaMemcpyAsync(&h_C[offset], &d_C[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Streams: %2d | Time: %8.3f ms | C[451][451] = %d\n", nStreams, milliseconds, h_C[451 * N + 451]);

    // Cleanup
    for (int i = 0; i < nStreams; i++) cudaStreamDestroy(streams[i]);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    int size = N * N * sizeof(int);
    
    // Allocate PINNED host memory for async operations
    int *h_A, *h_B, *h_C;
    cudaMallocHost((void**)&h_A, size);
    cudaMallocHost((void**)&h_B, size);
    cudaMallocHost((void**)&h_C, size);

    // Initialize matrices: A[i][j] = i, B[i][j] = j
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = i;
            h_B[i * N + j] = j;
        }
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    int test_streams[] = {1, 4, 16};
    
    for (int i = 0; i < 3; i++) {
        memset(h_C, 0, size);
        cudaMemset(d_C, 0, size);
        run_test(test_streams[i], h_A, h_B, h_C, d_A, d_B, d_C);
    }

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}