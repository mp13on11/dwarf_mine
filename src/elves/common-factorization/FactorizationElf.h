#pragma once

#include "common-factorization/BigInt.h"
#include "Elf.h"

#include <utility>

class FactorizationElf : public Elf
{
public:
    virtual void run(std::istream& input, std::ostream& output);

    virtual void stop() = 0;
    virtual std::pair<BigInt, BigInt> factorize(const BigInt& number) = 0;
};
