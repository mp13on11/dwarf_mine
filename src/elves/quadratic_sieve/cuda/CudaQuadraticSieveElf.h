#pragma once

#include "../QuadraticSieveElf.h"
#include <vector>

class CudaQuadraticSieveElf : public QuadraticSieveElf
{
public:
    virtual std::vector<BigInt> sieveSmoothSquares(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
    );
};
