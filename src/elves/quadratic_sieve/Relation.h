#pragma once

#include "common-factorization/BigInt.h"
#include "PrimeFactorization.h"

class Relation
{
public:
    Relation(const BigInt& a, const PrimeFactorization& factorization) :
        a(a),
        dependsOnPrime(0),
        primeFactorization(factorization),
        oddPrimePowers(factorization.oddPrimePowers())
    {}

    bool isPerfectCongruence() const;

public:
    BigInt a;
    smallPrime_t dependsOnPrime;
    PrimeFactorization primeFactorization;
    SparseVector<smallPrime_t> oddPrimePowers;
};

inline bool Relation::isPerfectCongruence() const
{
    return oddPrimePowers.empty();
}
