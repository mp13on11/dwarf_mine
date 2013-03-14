#pragma once

const int FIELD_DIMENSION = 8;

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
	//cassert(result < maximum, "Random %f - Maximum %lu = %f = %lu\n", random, maximum, random * maximum, result);
    return result;
} 