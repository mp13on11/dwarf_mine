#include <elves/cuda-utils/Memory.h>

class NumberHelper
{
public:
    static CudaUtils::Memory<uint32_t> BigIntToNumber(const BigInt& b)
    {
        NumData numberData;
        memset(numberData, 0, sizeof(uint32_t) * NUM_FIELDS);
        mpz_export(numberData, nullptr, -1, sizeof(uint32_t), 0, 0, b.get_mpz_t());

        CudaUtils::Memory<uint32_t> number_d(NUM_FIELDS);
        number_d.transferFrom(numberData);

        return number_d;
    }

    static BigInt NumberToBigInt(const CudaUtils::Memory<uint32_t>& number_d)
    {
        NumData outputData;
        number_d.transferTo(outputData);

        BigInt mpzResult;
        mpz_import(mpzResult.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, outputData);
        return mpzResult;
    }
};
