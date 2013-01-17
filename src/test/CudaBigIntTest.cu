#include <elves/factorize/cuda/kernel.cuh>
#include <elves/cuda-utils/ErrorHandling.h>

extern __global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output);

void testAdd(PNumData left, PNumData right, PNumData result)
{
    testAddKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testSub(PNumData left, PNumData right, PNumData result)
{
    testSubKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testMul(PNumData left, PNumData right, PNumData result)
{
    testMulKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testDiv(PNumData left, PNumData right, PNumData result)
{
    testDivKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}
