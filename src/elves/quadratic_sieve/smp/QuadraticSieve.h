#include <elves/common-factorization/BigInt.h>

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>
#include <list>

class Relation;
class PrimeFactorization;

typedef uint32_t smallPrime_t;
typedef std::vector<BigInt> SmoothSquares;
typedef std::vector<smallPrime_t> FactorBase;
typedef std::vector<Relation> Relations;

namespace QuadraticSieveHelper
{
    extern const std::pair<BigInt,BigInt> TRIVIAL_FACTORS;

    BigInt rootModPrime(const BigInt& n, const BigInt& primeMod);
    BigInt liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power);
    std::vector<BigInt> liftRoots(const std::vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower);
    std::vector<BigInt> squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power = 1);

    FactorBase createFactorBase(size_t numberOfPrimes);
    bool isNonTrivial(const std::pair<BigInt, BigInt>& pair, const BigInt& number);
    std::pair<BigInt,BigInt> factorsFromCongruence(const BigInt& a, const BigInt& b);

    std::pair<BigInt,BigInt> searchForRandomCongruence(const FactorBase& factorBase, const BigInt& number, size_t times);
    PrimeFactorization factorizeOverBase(const BigInt& number, const FactorBase& factorBase);
    void performGaussianElimination(Relations& relations);
}
/*
    std::pair<BigInt, BigInt> sieve();
    std::pair<BigInt, BigInt> sieveIntervalFast(const BigInt& start, const BigInt& end, size_t maxRelations);

    std::vector<BigInt> sieveSmoothSquares(const BigInt& start, const BigInt& end) const;
    */


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


