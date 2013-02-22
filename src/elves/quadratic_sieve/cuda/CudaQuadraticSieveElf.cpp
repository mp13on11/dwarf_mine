#include "CudaQuadraticSieveElf.h"
#include "Factorize.h"
#include "common/Utils.h"
#include "common-factorization/BigInt.h"
#include <cuda-utils/Memory.h>
#include "NumberHelper.h"
#include "KernelWrapper.h"

#include <array>
#include <algorithm>

using namespace std;

vector<BigInt> CudaQuadraticSieveElf::sieveSmoothSquares(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
)
{
    BigInt intervalLength = (end-start);

    size_t blockSize = intervalLength.get_ui();

    vector<uint32_t> logs(blockSize+1);
    BigInt x, remainder;
    //uint32_t logTreshold = (int)(lb(number));

    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % number;
        logs[i] = log_2_22(remainder);
    }

    CudaUtils::Memory<uint32_t> logs_d(logs.size());
    logs_d.transferFrom(logs.data());
    CudaUtils::Memory<uint32_t> factorBase_d(factorBase.size());
    factorBase_d.transferFrom(factorBase.data());

    array<uint32_t, 10> start_d;
	mpz_export((void*)start_d.data(), 0, -1, sizeof(uint32_t), 0, 0, start.get_mpz_t());

    megaWrapper(logs_d.get(), factorBase_d.get(), start_d.data(), blockSize);


//    CudaUtils::Memory<uint32_t> start_d = NumberHelper::BigIntToNumber(start);
//    CudaUtils::Memory<uint32_t> end_d = NumberHelper::BigIntToNumber(end);
//    CudaUtils::Memory<uint32_t> number_d = NumberHelper::BigIntToNumber(number);



    return vector<BigInt>();
}

/*

pair<BigInt, BigInt> CudaQuadraticSieveElf::factor(const BigInt& )
{

    // only for testing
    BigInt n("2");
    CudaUtils::Memory<uint32_t> n_d = NumberHelper::BigIntToNumber(n);
    //sieveIntervalWrapper(n_d.get(), nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr);


    //uint64_t value64 = value.getUint64Value();
    NumData result;
    CudaUtils::Memory<uint32_t> results_d(NUM_FIELDS);

    ::factorize(nullptr, results_d.get());
    results_d.transferTo(result);

    BigInt mpzResult;

    mpz_import(mpzResult.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, result);

    return make_pair(0, mpzResult);
}
*/
