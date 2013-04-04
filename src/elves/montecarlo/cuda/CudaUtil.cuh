#pragma once

#define FIELD_DIMENSION 8
#define FIELD_SIZE FIELD_DIMENSION * FIELD_DIMENSION


#include "CudaDebug.cuh"

__device__ size_t randomNumber(float* randomValues, size_t* randomSeed, size_t limit)
{
	size_t value = size_t(floor(randomValues[*randomSeed] * limit));
	if (value == limit)
	{
		--value;
	}	
	++(*randomSeed);
	return value;
}

/***
 * Delivers a random number x with 0 <= x < maximum
 */
__device__ size_t randomNumber(curandState* deviceStates, size_t maximum, float fakedRandom = -1)
{
	size_t threadGeneratorIndex = blockIdx.x;
	float random = 0;
	if (fakedRandom >= 0)
	{
		random = fakedRandom;
	}
	else
	{	
		curandState deviceState = deviceStates[threadGeneratorIndex];
		random = 1.0f - curand_uniform(&deviceState); // delivers (0, 1] - we need [0, 1)
		deviceStates[threadGeneratorIndex] = deviceState;
	}
	size_t result = size_t(floor(random * maximum));
	if (maximum == result) // nothing with (0,1] ... sometimes it is rounded to maximum - so we need to reduce manually
	{
		--result;
	}
	cassert(result < maximum, "Random %f - Maximum %lu = %f = %lu\n", random, maximum, random * maximum, result);
    return result;
} 

__device__ size_t numberOfMarkedFields(const bool* field)
{
	__shared__ unsigned int s[FIELD_DIMENSION];
	if (threadIdx.x % FIELD_DIMENSION == 0) s[threadIdx.x / FIELD_DIMENSION] = 0;
	__syncthreads();
	if (field[threadIdx.x]) atomicAdd(&s[threadIdx.x / FIELD_DIMENSION], 1u);
	__syncthreads();
	if (threadIdx.x % FIELD_DIMENSION == 0 && threadIdx.x != 0) atomicAdd(&s[0], s[threadIdx.x / FIELD_DIMENSION]);
	__syncthreads();
	
	return s[0];
}