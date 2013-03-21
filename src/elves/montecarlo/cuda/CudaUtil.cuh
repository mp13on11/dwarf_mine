#pragma once

const int FIELD_DIMENSION = 8;

#include "CudaDebug.cuh"

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
	__shared__ unsigned int s[8];
	if (threadIdx.x % 8 == 0) s[threadIdx.x / 8] = 0;
	__syncthreads();
	if (field[threadIdx.x]) atomicAdd(&s[threadIdx.x / 8], 1u);
	__syncthreads();
	if (threadIdx.x % 8 == 0 && threadIdx.x != 0) atomicAdd(&s[0], s[threadIdx.x / 8]);
	__syncthreads();
	return s[0];
}