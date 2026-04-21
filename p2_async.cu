#include <stdio.h>
#include <stdlib.h>

#define N 1024

__global__ void matmul(int *A, int *B, int *C, int row_offset) {
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
    int streamSize = (N / nStreams) * N; 
    int streamBytes = streamSize * sizeof(int);

    cudaStream_t streams[16];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(64 / nStreams, 64);

    cudaEventRecord(start);

    // --- PARALLEL ISSUE LOOPS ---
    
    // Loop 1: All Host-to-Device Copies
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_A[offset], &h_A[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    }

    // Loop 2: All Kernel Launches
    for (int i = 0; i < nStreams; i++) {
        int row_offset = i * (N / nStreams);
        matmul<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A, d_B, d_C, row_offset);
    }

    // Loop 3: All Device-to-Host Copies
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&h_C[offset], &d_C[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Streams: %2d | Time: %8.3f ms | C[451][451] = %d\n", nStreams, milliseconds, h_C[451 * N + 451]);

    for (int i = 0; i < nStreams; i++) cudaStreamDestroy(streams[i]);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    int size = N * N * sizeof(int);
    
    int *h_A, *h_B, *h_C;
    cudaMallocHost((void**)&h_A, size);
    cudaMallocHost((void**)&h_B, size);
    cudaMallocHost((void**)&h_C, size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = i;
            h_B[i * N + j] = j;
        }
    }

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

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}