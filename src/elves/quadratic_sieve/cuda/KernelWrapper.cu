#include "kernel.cuh"
#include "Number.cuh"
#include "Factorize.h"
#include "common/Utils.h"
#include "stdio.h"

const size_t BLOCK_SIZE = 64;

void megaWrapper(const uint32_t* number, uint32_t* logs_d, const uint32_t* factorBase_d, const size_t factorBaseSize, const uint32_t* start, const uint32_t* end, const uint32_t intervalLength)
{
    size_t numThreads = div_ceil(intervalLength, (uint32_t)NUMBERS_PER_THREAD);
    size_t numBlocks = div_ceil(numThreads, BLOCK_SIZE);
    
    Number* number_d; 
    cudaMalloc(&number_d, sizeof(uint32_t)*NUM_FIELDS);
    cudaMemcpy(number_d, number, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
       
	Number* start_d; 
	cudaMalloc(&start_d, sizeof(uint32_t)*NUM_FIELDS);
	cudaMemcpy(start_d, start, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	Number* end_d; 
	cudaMalloc(&end_d, sizeof(uint32_t)*NUM_FIELDS);
	cudaMemcpy(end_d, end, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	//printf("garbage2? %d", end[9]);
	
	printf("before megaKernel in wrapper, numBlocks: %d, numThreads: %d\n", numBlocks, numThreads);
	megaKernel<<<numBlocks, BLOCK_SIZE>>>(number_d, logs_d, factorBase_d, (int)factorBaseSize, start_d, end_d, intervalLength);
	printf("after megaKernel\n");
	
	cudaDeviceSynchronize();
	
	cudaFree(start_d);
	cudaFree(end_d);
	cudaFree(number_d);
}
