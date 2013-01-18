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
        memcpy(output, outputNum.fields, sizeof(uint32_t)*NUM_FIELDS);
    }
}

#define TEST_OP(op)                                             \
    Number left(pLeft);                                         \
    Number right(pRight);                                       \
    Number result(left op right);                             \
    memcpy(output, result.fields, sizeof(uint32_t)*NUM_FIELDS);

__global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    TEST_OP(+)
}


__global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    TEST_OP(-)
}

__global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    TEST_OP(*)
}

__global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    TEST_OP(/)
}

__global__ void testSmallerThanKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    bool result(left < right);
    memcpy(output, &result, sizeof(bool));
}


__global__ void testShiftLeftKernel(PNumData pLeft, uint32_t offset, PNumData output)
{
    Number left(pLeft);
    Number result(left << offset);
    memcpy(output, &result,  sizeof(uint32_t)*NUM_FIELDS);
}
