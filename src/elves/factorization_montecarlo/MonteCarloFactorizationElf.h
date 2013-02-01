#pragma once

#include "common-factorization/FactorizationElf.h"

class MonteCarloFactorizationElf : public FactorizationElf
{
public:
    bool finished;

    MonteCarloFactorizationElf();

    virtual std::pair<BigInt, BigInt> factor(const BigInt& number);
    void stop();

    size_t randomSeed() const;
};
