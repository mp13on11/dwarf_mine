#pragma once

#include "FactorizationElf.h"

class SmpFactorizationElf : public FactorizationElf
{
public:
    bool finished;

    SmpFactorizationElf();

    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
    virtual void stop();

    size_t randomSeed() const;
};
