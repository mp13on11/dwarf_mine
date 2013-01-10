#pragma once

#include "elves/factorize/FactorizationElf.h"

class SmpFactorizationElf : public FactorizationElf
{
public:
    bool finished;

    SmpFactorizationElf();

    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
    size_t randomSeed() const;
};
