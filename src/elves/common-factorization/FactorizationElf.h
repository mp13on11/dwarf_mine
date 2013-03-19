#pragma once

#include "common-factorization/BigInt.h"
#include "Elf.h"

#include <utility>

class FactorizationElf : public Elf
{
public:
    virtual void stop() = 0;
    virtual std::pair<BigInt, BigInt> factor(const BigInt& number) = 0;
};
