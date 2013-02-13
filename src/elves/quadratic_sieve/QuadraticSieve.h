#pragma once

#include "common-factorization/BigInt.h"
#include "Types.h"
#include "Relation.h"

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>
#include <list>

class PrimeFactorization;

typedef std::vector<BigInt> SmoothSquares;
typedef std::vector<smallPrime_t> FactorBase;
typedef std::vector<Relation> Relations;

namespace QuadraticSieveHelper
{
	typedef std::function<std::pair<BigInt, BigInt>(std::vector<Relation>&, const FactorBase&, const BigInt&)> SieveCallback;

    extern const std::pair<BigInt,BigInt> TRIVIAL_FACTORS;

    BigInt rootModPrime(const BigInt& n, const BigInt& primeMod);
    BigInt liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power);
    std::vector<BigInt> liftRoots(const std::vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower);
    std::vector<BigInt> squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power = 1);

    FactorBase createFactorBase(size_t numberOfPrimes);
    bool isNonTrivial(const std::pair<BigInt, BigInt>& pair, const BigInt& number);
    std::pair<BigInt,BigInt> factorsFromCongruence(const BigInt& a, const BigInt& b, const BigInt& number);

    std::pair<BigInt,BigInt> searchForRandomCongruence(const FactorBase& factorBase, const BigInt& number, size_t times, const Relations& relations);
    PrimeFactorization factorizeOverBase(const BigInt& number, const FactorBase& factorBase);
    void performGaussianElimination(Relations& relations);
    std::pair<BigInt, BigInt> factor(const BigInt& number, SieveCallback sieveCallback);
}
