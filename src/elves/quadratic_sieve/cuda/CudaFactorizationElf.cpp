#include "CudaFactorizationElf.h"
#include "Factorize.h"
#include <cuda-utils/Memory.h>

#include <array>
#include <algorithm>

using namespace std;

pair<BigInt, BigInt> CudaFactorizationElf::factor(const BigInt& )
{
    //uint64_t value64 = value.getUint64Value();
    NumData result;
    CudaUtils::Memory<uint32_t> results_d(NUM_FIELDS);

    ::factorize(nullptr, results_d.get());
    results_d.transferTo(result);

    BigInt mpzResult;

    mpz_import(mpzResult.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, result);

    return make_pair(0, mpzResult);
}
