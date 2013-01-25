#include <cuda.h>
#include "Number.cuh"
#include <iostream>

__global__ void sieveIntervalKernel(PNumData pn, uint32_t* logs, uint32_t* rootsModPrime, uint32_t* factorBase, int factorBaseSize, PNumData pStart, PNumData pEnd);

void sieveIntervalWrapper(PNumData pn, uint32_t* logs, uint32_t* rootsModPrime, uint32_t* factorBase, int factorBaseSize, PNumData pStart, PNumData pEnd)
{
    int threadsPerBlock = 512;
    int numberOfBlocks = 1 + ((factorBaseSize - 1) / threadsPerBlock);
    std::cout << "Blocks: " << numberOfBlocks << " Threads/Block: " << threadsPerBlock << std::endl;
    sieveIntervalKernel<<<numberOfBlocks, threadsPerBlock>>>(pn, logs, rootsModPrime, factorBase, factorBaseSize, pStart, pEnd);
}
