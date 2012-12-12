#include "MatrixMultiplication.h"
#include "ErrorHandling.h"
#include <cuda.h>
#include <iostream>

//kernel declaration
__global__ void gemmKernel(int m, int n, int k, float* left, float* right, float* out);

int div_ceil(int a, int b)
{
    int result = a/b;
    if (a % b != 0) ++result;
    return result;
}
//kernel calling function
void gemm(int m, int n, int k, float* left, float* right, float* out, int blockSize)
{
	using namespace std;

    dim3 dimGrid(div_ceil(n, blockSize), div_ceil(m, blockSize));
    dim3 dimBlock(blockSize, blockSize);
    cout << m << ", " << n << endl;
    cout << div_ceil(m, blockSize) << ", " << div_ceil(n, blockSize) << endl;
    gemmKernel <<< dimGrid, dimBlock >>>(m, n, k, left, right, out);
    CudaUtils::checkState();
}
