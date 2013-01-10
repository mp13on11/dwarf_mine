#pragma once

#include "elves/factorize/FactorizationElf.h"

class SmpFactorizationElf : public FactorizationElf
{
public:
    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number);
};
