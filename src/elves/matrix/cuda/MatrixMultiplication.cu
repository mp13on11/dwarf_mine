#include "MatrixMultiplication.h"
#include "ErrorHandling.h"
#include <main/Utils.h>

#include <cuda.h>
#include <iostream>

//kernel declaration
__global__ void gemmKernel(int m, int n, int k, float* left, float* right, float* out);

//kernel calling function
void gemm(int m, int n, int k, float* left, float* right, float* out, int blockSize)
{
	using namespace std;

    dim3 dimGrid(div_ceil(n, blockSize), div_ceil(m, blockSize));
    dim3 dimBlock(blockSize, blockSize);
    gemmKernel <<< dimGrid, dimBlock >>>(m, n, k, left, right, out);
    CudaUtils::checkState();
    //cudaDeviceReset();
}
