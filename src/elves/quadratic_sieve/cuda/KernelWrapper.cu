#include "kernel.cuh"
#include "Number.cuh"
#include "Factorize.h"
#include "common/Utils.h"

const size_t BLOCK_SIZE = 64;

void megaWrapper(uint32_t* logs_d, const uint32_t* factorBase_d, const uint32_t* start, const uint32_t intervalLength)
{
    size_t numThreads = div_ceil(intervalLength, NUMBERS_PER_THREAD);
    size_t numBlocks = div_ceil(numThreads, BLOCK_SIZE);
	Number* start_d; 
	cudaMalloc(&start_d, sizeof(uint32_t)*NUM_FIELDS);
	cudaMemcpy(start_d, start, 10, cudaMemcpyHostToDevice);
	
	megaKernel<<<numBlocks, BLOCK_SIZE>>>(logs_d, factorBase_d, start_d, intervalLength);
	cudaFree(start_d);
}