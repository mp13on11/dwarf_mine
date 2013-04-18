#pragma once

#include "common-factorization/BigInt.h"
#include <Elf.h>
#include "QuadraticSieveHelper.h"

class Relation;

class QuadraticSieveElf : public Elf
{
public:
    virtual std::vector<BigInt> sieveSmoothSquares(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
    ) = 0;
};
