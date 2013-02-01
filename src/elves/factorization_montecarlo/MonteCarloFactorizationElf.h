#pragma once

#include "common-factorization/FactorizationElf.h"

class MonteCarloFactorizationElf : public FactorizationElf
{
public:
    bool finished;

    MonteCarloFactorizationElf();

    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
    virtual void stop();

    size_t randomSeed() const;
};
