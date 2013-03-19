#pragma once

#include "elves/common-factorization/BigInt.h"
#include "MonteCarloFactorizationElf.h"

#include <functional>
#include <map>

class SequentialFactorizer
{
public:
    SequentialFactorizer(const BigInt& m, const MonteCarloFactorizationElf& elf);

    void run();
    std::pair<BigInt, BigInt> result() const;

private:
    const MonteCarloFactorizationElf& elf;
    BigInt m, p, q;
    std::multimap<BigInt, BigInt> remainders;
    gmp_randclass generator;

    typedef std::multimap<BigInt, BigInt>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

    BigInt generateRandomNumberSmallerThan(const BigInt& n);
};

inline std::pair<BigInt, BigInt> SequentialFactorizer::result() const
{
    return std::pair<BigInt, BigInt>(p, q);
}
