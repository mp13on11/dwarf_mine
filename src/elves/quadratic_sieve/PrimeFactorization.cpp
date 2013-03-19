#include "PrimeFactorization.h"
#include <iostream>

using namespace std;

bool PrimeFactorization::empty() const
{
    return primePowers.empty();
}

PrimeFactorization PrimeFactorization::sqrt() const
{
    PrimeFactorization result;
    for(auto primeEntry : primePowers)
    {
        result.primePowers.push_back({primeEntry.first, primeEntry.second / 2});
    }
    return result;
}

PrimeFactorization PrimeFactorization::combine(const PrimeFactorization& other) const
{
    PrimeFactorization result;
    auto aIt=primePowers.begin();
    auto bIt=other.primePowers.begin();
    for(; aIt != primePowers.end() && bIt != other.primePowers.end(); )
    {
        if(aIt->first < bIt->first)
        {
            result.primePowers.push_back(*aIt);
            aIt++;
            continue;
        }
        if(aIt->first > bIt->first)
        {
            result.primePowers.push_back(*bIt);
            bIt++;
            continue;
        }
        result.primePowers.push_back({aIt->first, aIt->second + bIt->second});
        aIt++;
        bIt++;
    }

    result.primePowers.insert(result.primePowers.end(), aIt, primePowers.end());
    result.primePowers.insert(result.primePowers.end(), bIt, other.primePowers.end());

    return result;
}

BigInt PrimeFactorization::multiply() const
{
    BigInt result(1);
    BigInt prime;
    BigInt primePower;
    for(auto primeEntry : primePowers)
    {
        prime = primeEntry.first;
        mpz_pow_ui(primePower.get_mpz_t(), prime.get_mpz_t(), primeEntry.second);
        result *= primePower;
    }
    return result;
}

SparseVector<smallPrime_t> PrimeFactorization::oddPrimePowers() const
{
    SparseVector<smallPrime_t> result;
    for(auto primeEntry : primePowers)
    {
        if(primeEntry.second % 2 == 1)
            result.indices.push_back(primeEntry.first);
    }
    return result;
}

void PrimeFactorization::print(ostream& stream) const
{
    bool first = true;
    for(const auto& pairy : primePowers)
    {
        if(first)
            first = false;
        else
            stream << " * ";
        stream << pairy.first;
        if(pairy.second > 1)
            stream << "^" << pairy.second;
    }
}

void PrimeFactorization::add(const smallPrime_t& prime, uint32_t power)
{
    primePowers.push_back({prime, power});
}

ostream& operator<<(ostream& stream, const PrimeFactorization& factorization)
{
    factorization.print(stream);
    return stream;
}
