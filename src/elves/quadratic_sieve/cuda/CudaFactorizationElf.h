#pragma once

#include <elves/factorize/FactorizationElf.h>
#include <vector>

class CudaFactorizationElf : public FactorizationElf
{
public:
    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
    virtual void stop();
};
