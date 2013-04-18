#pragma once

#include "Number.cuh"

__global__ void megaKernel(const Number* number, uint32_t* logs, const uint32_t* factorBase, const int factorBaseSize, const Number* start, const Number* end, const uint32_t intervalLength);
