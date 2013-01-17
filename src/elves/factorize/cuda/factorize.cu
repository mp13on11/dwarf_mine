#include "kernel.cuh"
#include <elves/cuda-utils/ErrorHandling.h>
#include <cmath>

const unsigned int BLOCK_SIZE = 64;

unsigned int determineBlockSize(uint64_t amount)
{
    return 1 + (amount - 1) / BLOCK_SIZE;
}

void factorize(PNumData input, PNumData output)
{
    factorizeKernel<<<1, 32>>>(input, output);
    CudaUtils::checkState();
}
