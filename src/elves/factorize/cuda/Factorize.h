#pragma once

#ifdef __CUDACC__
typedef unsigned long uint64_t;
typedef unsigned int uint32_t;
#else
#include <cstdint>
#endif

const int NUM_FIELDS = 10;

typedef uint32_t NumData[NUM_FIELDS];
typedef uint32_t* PNumData;

extern void factorize(PNumData input, PNumData output);
