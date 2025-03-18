// LeetGPU - Vector Addition

#include <cuda_runtime.h>
#include <iostream>

#define ROWS 2
#define COLS 5

__global__ void matrix_add(const float *A, const float *B, float *C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}

void solve(const float *A, const float *B, float *C, int rows, int cols)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}

int main()
{
    const int N = ROWS * COLS;
    float h_A[ROWS][COLS] = {{1, 2, 3, 4, 5},
                             {6, 7, 8, 9, 10}};

    float h_B[ROWS][COLS] = {{2, 4, 6, 8, 10},
                             {12, 14, 16, 18, 20}};

    float h_C[ROWS][COLS];

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, ROWS, COLS);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            std::cout << h_C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
