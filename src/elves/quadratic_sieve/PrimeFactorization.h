#pragma once

#include "Types.h"
#include "common-factorization/BigInt.h"
#include "SparseVector.h"
#include <list>
#include <utility>
#include <iosfwd>

class PrimeFactorization
{
public:
    bool empty() const;
    PrimeFactorization combine(const PrimeFactorization& other) const;
    PrimeFactorization sqrt() const;
    BigInt multiply() const;
    SparseVector<smallPrime_t> oddPrimePowers() const;
    void add(const smallPrime_t& prime, uint32_t power);

    void print(std::ostream& stream) const;

private:
    std::list<std::pair<smallPrime_t, uint32_t>> primePowers;
};

inline PrimeFactorization sqrt(const PrimeFactorization& p)
{
    return p.sqrt();
}

extern std::ostream& operator<<(std::ostream& stream, const PrimeFactorization& factorization);
