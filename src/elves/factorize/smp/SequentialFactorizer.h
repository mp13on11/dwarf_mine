#pragma once

#include "elves/factorize/BigInt.h"
#include "elves/factorize/smp/SmpFactorizationElf.h"

#include <functional>
#include <map>
#include <random>

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
    std::uniform_int_distribution<uint32_t> distribution;
    std::mt19937 engine;
    std::function<uint32_t()> generator;

    typedef std::multimap<BigInt, BigInt>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

    BigInt generateRandomNumberSmallerThan(const BigInt& n) const;
};

inline std::pair<BigInt, BigInt> SequentialFactorizer::result() const
{
    return std::pair<BigInt, BigInt>(p, q);
}
