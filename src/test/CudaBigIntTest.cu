#include <elves/quadratic_sieve/cuda/kernel.cuh>
#include <elves/cuda-utils/ErrorHandling.h>

extern __global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testModKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testSmallerThanKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testLargerThanKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testLargerEqualKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testEqualKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testShiftLeftKernel(PNumData pLeft, uint32_t offset, PNumData output);
extern __global__ void testShiftRightKernel(PNumData pLeft, uint32_t offset, PNumData output);
extern __global__ void testModPowKernel(PNumData base, PNumData exponent, PNumData mod, PNumData result);

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

void testMod(PNumData left, PNumData right, PNumData result)
{
    testModKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testSmallerThan(PNumData left, PNumData right, bool* result)
{
    testSmallerThanKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testLargerThan(PNumData left, PNumData right, bool* result)
{
    testLargerThanKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testLargerEqual(PNumData left, PNumData right, bool* result)
{
    testLargerEqualKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testEqual(PNumData left, PNumData right, bool* result)
{
    testEqualKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testShiftLeft(PNumData left,uint32_t offset, PNumData result)
{
    testShiftLeftKernel<<<1, 1>>>(left, offset, result);
    CudaUtils::checkState();
}

void testShiftRight(PNumData left,uint32_t offset, PNumData result)
{
    testShiftRightKernel<<<1, 1>>>(left, offset, result);
    CudaUtils::checkState();
}

void testModPow(PNumData base, PNumData exponent, PNumData mod, PNumData result)
{
    testModPowKernel<<<1, 1>>>(base, exponent, mod, result);
    CudaUtils::checkState();
}
