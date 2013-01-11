#pragma once

#include <elves/factorize/FactorizationElf.h>

class CudaFactorizationElf : public FactorizationElf
{
public:
    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
    virtual void stop();
};
