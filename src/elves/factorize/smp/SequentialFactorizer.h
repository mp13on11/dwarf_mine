#pragma once

#include "elves/factorize/BigInt.h"
#include "elves/factorize/smp/SmpFactorizationElf.h"

#include <functional>
#include <map>

class SequentialFactorizer
{
public:
    SequentialFactorizer(const BigInt& m, const SmpFactorizationElf& elf);

    void run();
    std::pair<BigInt, BigInt> result() const;

private:
    const SmpFactorizationElf& elf;
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
