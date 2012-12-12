__global__ void gemmKernel(int m, int n, int k, float* left, float* right, float* out)
{
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (column < n && row < m)
    {
        float sum=0; 
        for (int i=0; i < k; ++i)
        {
            sum += left[row*k+i] * right[n*i+column];
        } 

        out[row*n+column] = sum;
    }
}
