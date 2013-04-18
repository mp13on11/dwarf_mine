#pragma once

#ifdef __CUDACC__
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
#else
#include <cstdint>
#endif

typedef uint32_t FieldType;

const int NUM_FIELDS = 10;
const size_t DATA_SIZE_BYTES = sizeof(FieldType)*NUM_FIELDS;

typedef FieldType NumData[NUM_FIELDS];
typedef FieldType* PNumData;

extern void factorize(PNumData input, PNumData output);
