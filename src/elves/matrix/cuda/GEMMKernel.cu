#include <stdio.h>
#include <assert.h>
#include "MatrixMultiplication.h"

const size_t BLOCK_SIZE = DEFAULT_BLOCK_SIZE;

__device__ int div_ceil_d(int x, int y)
{
//    return 1 + ((x - 1) / y);
//    return (x + y - 1) / y;

    return (x % y) ? x / y + 1 : x / y;
}

struct Matrix
{
    int cols;
    int rows;
    int stride;
    float* data;
};

__device__ void setElement(Matrix m, int row, int col, float value)
{
    if (row >= m.rows || col >= m.cols) return;
    m.data[(m.stride * row) + col] = value;
}

__device__ float getElement(Matrix m, int row, int col)
{
    if (row >= m.rows || col >= m.cols) return 0;
    return m.data[(m.stride * row) + col];
}

__device__ Matrix getSubMatrix(Matrix m, int blockRow, int blockColumn)
{
    Matrix n;
    n.rows = ((blockRow+1)*blockDim.x > m.rows) ? (m.rows - blockRow*blockDim.x) : blockDim.x;
    n.cols = ((blockColumn+1)*blockDim.x > m.cols) ? (m.cols - blockColumn*blockDim.x) : blockDim.x;
    n.stride = m.stride;
    n.data = &m.data[blockRow * m.stride * blockDim.x + blockColumn * blockDim.x];
    return n;
}


__global__ void gemmKernel(int m, int n, int k, float* left, float* right, float* out)
{
    Matrix leftMatrix;
    leftMatrix.rows = m;
    leftMatrix.cols = k;
    leftMatrix.stride = k;
    leftMatrix.data = left;

    Matrix rightMatrix;
    rightMatrix.rows = k;
    rightMatrix.cols = n;
    rightMatrix.stride = n;
    rightMatrix.data = right;

    Matrix outMatrix;
    outMatrix.rows = m;
    outMatrix.cols = n;
    outMatrix.stride = n;
    outMatrix.data = out;

    int blockRow = blockIdx.y;
    int blockColumn = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    Matrix outSub = getSubMatrix(outMatrix, blockRow, blockColumn);

    float sum = 0.0f;

    for (int block = 0, end= div_ceil_d(leftMatrix.cols, blockDim.x); block < end ; ++block)
    {
        __shared__ float leftSub_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float rightSub_s[BLOCK_SIZE][BLOCK_SIZE];

        Matrix leftSub = getSubMatrix(leftMatrix, blockRow, block);
        Matrix rightSub = getSubMatrix(rightMatrix, block, blockColumn);

        leftSub_s[row][col] = getElement(leftSub, row, col);
        rightSub_s[row][col] = getElement(rightSub, row, col);

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i)
        {
            sum += leftSub_s[row][i] * rightSub_s[i][col];
        }
    }

    setElement(outSub, row, col, sum);
}
