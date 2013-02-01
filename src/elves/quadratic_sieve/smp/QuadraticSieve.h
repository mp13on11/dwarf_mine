#include "common-factorization/BigInt.h"

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>
#include <list>
#include <random>

class Relation;
class PrimeFactorization;

typedef uint32_t smallPrime_t;

class QuadraticSieve
{
public:
    QuadraticSieve(const BigInt& n) :
        n(n), binaryDistribution(0,1)
    {}
    std::pair<BigInt, BigInt> factorize();
    static BigInt rootModPrime(const BigInt& n, const BigInt& primeMod);
    static BigInt liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power);
    static std::vector<BigInt> liftRoots(const std::vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower);

    static std::vector<BigInt> squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power = 1);

private:
    void createFactorBase(size_t numberOfPrimes);
    std::pair<BigInt, BigInt> sieve();
    std::pair<BigInt, BigInt> sieveIntervalFast(const BigInt& start, const BigInt& end, size_t maxRelations);
    bool isNonTrivial(const std::pair<BigInt, BigInt>& pair) const;
    std::pair<BigInt,BigInt> factorsFromCongruence(const BigInt& a, const BigInt& b) const;
    void performGaussianElimination();
    std::pair<BigInt,BigInt> searchForRandomCongruence(size_t times) const;
    std::pair<BigInt,BigInt> pickRandomCongruence() const;
    PrimeFactorization factorizeOverBase(const BigInt& x) const;
    std::vector<BigInt> sieveSmoothSquares(const BigInt& start, const BigInt& end) const;


    void print(const Relation& r) const;

    BigInt n;
    std::vector<smallPrime_t> factorBase;
    std::vector<Relation> relations;

    mutable std::uniform_int_distribution<int> binaryDistribution;
    mutable std::mt19937 randomEngine;

    static const std::pair<BigInt,BigInt> TRIVIAL_FACTORS;
};

template<typename T>
class SparseVector
{
public:
    bool empty() const {return indices.empty();}
    void isSet(T index) const;
    void set(T index);
    void flip(T index);
    std::vector<T> indices;
};

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

inline std::ostream& operator<<(std::ostream& stream, const PrimeFactorization& factorization)
{
    factorization.print(stream);
    return stream;
}

class Relation
{
public:
    Relation(const BigInt& a, const PrimeFactorization& factorization) :
        a(a), dependsOnPrime(0), primeFactorization(factorization), oddPrimePowers(factorization.oddPrimePowers())
    {}
    bool isPerfectCongruence() const;

public:
    BigInt a;
    smallPrime_t dependsOnPrime;
    PrimeFactorization primeFactorization;
    SparseVector<smallPrime_t> oddPrimePowers;
};

PrimeFactorization sqrt(const PrimeFactorization& p);


