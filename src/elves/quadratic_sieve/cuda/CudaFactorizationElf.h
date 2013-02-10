#pragma once

#include <elves/common-factorization/FactorizationElf.h>
#include <vector>

class CudaFactorizationElf : public FactorizationElf
{
public:
    virtual std::pair<BigInt, BigInt> factor(const BigInt& number);
};
