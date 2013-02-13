#pragma once

#include "common-factorization/BigInt.h"
#include <Elf.h>
#include "QuadraticSieve.h"

class Relation;

class QuadraticSieveElf : public Elf
{
public:
    virtual std::pair<BigInt, BigInt> sieve(std::vector<Relation>& relations, const FactorBase& factorBase, const BigInt& number) = 0;
};
