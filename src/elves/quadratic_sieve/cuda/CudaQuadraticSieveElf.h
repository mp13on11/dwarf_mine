#pragma once

#include "../QuadraticSieveElf.h"
#include <vector>

class CudaQuadraticSieveElf : public QuadraticSieveElf
{
public:
    virtual std::pair<BigInt, BigInt> sieve(std::vector<Relation>& relations, const FactorBase& factorBase, const BigInt& number);
};
