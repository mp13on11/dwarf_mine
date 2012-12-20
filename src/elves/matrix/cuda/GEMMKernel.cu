#define MYASSERT(cond) \
    if (!(cond)) { \
        errorflag=1; \
        __threadfence(); \
        asm("trap;"); \
    } \

__device__ int errorflag = 0;

__device__ int div_ceil_d(int x, int y)
{
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
    m.data[(m.stride * row) + col] = value;  
}

__device__ float getElement(Matrix m, int row, int col)
{
    return m.data[(m.stride * row) + col];
}

__device__ Matrix getSubMatrix(Matrix m, int blockRow, int blockColumn)
{
    Matrix n;
    n.rows = blockDim.x;
    n.cols = blockDim.x;
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

    MYASSERT(blockDim.y == 32);

    for (int block = 0; block < div_ceil_d(leftMatrix.cols, blockDim.x); ++block) 
    {
        __shared__ float leftSub_s[32*32];
        __shared__ float rightSub_s[32*32];

        Matrix leftSub = getSubMatrix(leftMatrix, blockRow, block);
        Matrix rightSub = getSubMatrix(rightMatrix, block, blockColumn);

        
        leftSub_s[row * blockDim.x + col] = getElement(leftSub, row, col);
        rightSub_s[row * blockDim.x + col] = getElement(rightSub, row, col);

        
        __syncthreads();
        
        for (int i = 0; i < blockDim.x; ++i) 
        {
            sum += leftSub_s[row * blockDim.x + i] * rightSub_s[i * blockDim.x + col];
        }
        __syncthreads();
    }    

    setElement(outSub, row, col, sum);

}
