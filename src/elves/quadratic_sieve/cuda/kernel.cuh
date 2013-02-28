#pragma once

#include "Number.cuh"

#define NUMBERS_PER_THREAD 32

__global__ void megaKernel(uint32_t* logs, const uint32_t* factorBase, const int factorBaseSize, const Number* start, const Number* end, const uint32_t intervalLength);
