#pragma once

#include "../QuadraticSieveElf.h"

class SmpQuadraticSieveElf : public QuadraticSieveElf
{
public:
    virtual std::vector<BigInt> sieveSmoothSquares(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
    );
};
