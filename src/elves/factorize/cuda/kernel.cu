#include "kernel.cuh"

__device__ const NumData exampleData = { 0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0, 0, 0 };
__device__ const NumData example2 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

__global__ void factorizeKernel(PNumData input, PNumData output)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index == 0)
    {
        Number test(exampleData);
        Number test2(example2);
        Number outputNum(test - test2);
        outputNum.writeTo(output);
    }
}

__global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left + right);
    result.writeTo(output);
}


__global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left - right);
    result.writeTo(output);
}

__global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left * right);
    result.writeTo(output);
}

__global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left / right);
    result.writeTo(output);
}

__global__ void testSmallerThanKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    *output = left < right;
}


__global__ void testShiftLeftKernel(PNumData pLeft, uint32_t offset, PNumData output)
{
    Number left(pLeft);
    Number result(left << offset);
    result.writeTo(output);
}

__global__ void testShiftRightKernel(PNumData pLeft, uint32_t offset, PNumData output)
{
    Number left(pLeft);
    Number result(left >> offset);
    result.writeTo(output);
}
