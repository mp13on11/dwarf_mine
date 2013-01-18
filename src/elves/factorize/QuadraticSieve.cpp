#include "QuadraticSieve.h"
#include <algorithm>
#include <future>


using namespace std;

const pair<BigInt,BigInt> QuadraticSieve::TRIVIAL_FACTORS(0,0);

pair<BigInt, BigInt> QuadraticSieve::factorize()
{
    pair<BigInt, BigInt> factors = TRIVIAL_FACTORS;

    createFactorBase(50);


    // sieve
    cout << "sieving relations ..." << endl; 
    factors = sieve();
    if(isNonTrivial(factors))
        return factors;

    // bring relations into lower diagonal form
    cout << "perform gaussian elimination ..." << endl;
    performGaussianElimination();


    cout << "combining random congruences ..." << endl;
    factors = searchForRandomCongruence(100);

    return factors;
}

pair<BigInt,BigInt> QuadraticSieve::factorsFromCongruence(const BigInt& a, const BigInt& b) const
{
    BigInt sum = a+b;
    BigInt diff = (a>b)?(a-b):(b-a);
    BigInt p, q;
    mpz_gcd(p.get_mpz_t(), sum.get_mpz_t(), n.get_mpz_t());
    mpz_gcd(q.get_mpz_t(), diff.get_mpz_t(), n.get_mpz_t());
    return pair<BigInt, BigInt>(p, q);
}

bool QuadraticSieve::isNonTrivial(const pair<BigInt,BigInt>& factors) const
{
    const BigInt& p = factors.first;
    const BigInt& q = factors.second;

    return p>1 && p<n && q>1 && q<n;
}


pair<BigInt, BigInt> QuadraticSieve::sieve()
{
    BigInt intervalStart = sqrt(n)+1;
    BigInt intervalEnd = sqrt(n)+1 + BigInt("10000000");
    return sieveInterval(intervalStart, intervalEnd, factorBase.size() + 20);
}


pair<BigInt, BigInt> QuadraticSieve::sieveInterval(const BigInt& start, const BigInt& end, size_t maxRelations)
{
    BigInt remainder;

    for(BigInt x = start; x < end; x++)
    {
        remainder = (x*x) % n;
   
        PrimeFactorization factorization = factorizeOverBase(remainder);
        if(factorization.empty())
            continue;


        Relation relation(x, factorization);
        //cout << "NEW: ", print(relation);
        if(relation.isPerfectCongruence())
        {
            auto factors = factorsFromCongruence(x, sqrt(factorization).multiply());
            if(isNonTrivial(factors))
            {
                continue;
                return factors;
            }
        }       
        
        relations.push_back(relation);
        
        if(relations.size() > maxRelations)
            break;

    }

    return TRIVIAL_FACTORS;
}


void QuadraticSieve::print(const Relation& r) const
{
    cout << r.a << "^2%%N=\t";
    for(const smallPrime_t& prime : factorBase)
    {
       cout << ((find(r.oddPrimePowers.indices.begin(), r.oddPrimePowers.indices.end(), prime) == r.oddPrimePowers.indices.end()) ? 0 : 1); 
    }
    cout << " (";
    for(uint32_t p : r.oddPrimePowers.indices)
    {
        cout << p << " ";
    }
    cout << ")";

    /*cout << " [";
    for(uint32_t p : r.primeFactorization)
    {
        cout << p << " ";
    }
    cout << "]";
    */
    cout << endl;
}


pair<BigInt,BigInt> QuadraticSieve::searchForRandomCongruence(size_t times) const
{
    for(size_t i=0; i<times; i++)
    {
        pair<BigInt, BigInt> result = pickRandomCongruence();
        if(isNonTrivial(result))
            return result;
    }
    return TRIVIAL_FACTORS;
}

pair<BigInt,BigInt> QuadraticSieve::pickRandomCongruence() const
{
    vector<bool> primeMask(factorBase.back()+1);

    vector<Relation> selectedRelations;
    for(auto relIter = relations.rbegin(); relIter != relations.rend(); relIter++)
    {
        const Relation& relation = *relIter;

        if((relation.dependsOnPrime == 0 &&  binaryDistribution(randomEngine) == 1) || (relation.dependsOnPrime > 0 && primeMask[relation.dependsOnPrime]))
        {   
            selectedRelations.push_back(relation);
            for(uint32_t p : relation.oddPrimePowers.indices)
                primeMask[p] = !primeMask[p];
        }
    }
    
    if(selectedRelations.empty())
        return TRIVIAL_FACTORS;



    BigInt a = 1;
    PrimeFactorization factorization;
    for(const Relation& relation : selectedRelations)
    {
        a = (a * relation.a) % n;
        factorization = factorization.combine(relation.primeFactorization);
    }

    BigInt b = factorization.sqrt().multiply();

    return factorsFromCongruence(a,b);
}



typedef struct {
    bool operator() (const Relation& a, const Relation& b) 
    { 
        auto aStart = upper_bound(a.oddPrimePowers.indices.begin(), a.oddPrimePowers.indices.end(), minPrime);
        auto bStart = upper_bound(b.oddPrimePowers.indices.begin(), b.oddPrimePowers.indices.end(), minPrime);

        auto aIt=aStart, bIt=bStart;
        for(aIt=aStart, bIt=bStart; aIt != a.oddPrimePowers.indices.end() && bIt != b.oddPrimePowers.indices.end(); aIt++, bIt++)
        {
            if(*aIt < *bIt)
                return true;
            else if (*aIt > *bIt)
                return false;
        }
        return aIt != a.oddPrimePowers.indices.end();
    }
    uint32_t minPrime;
} RelationComparator;


void QuadraticSieve::performGaussianElimination()
{
    RelationComparator comparator;
    uint32_t currentPrime = 0;

    for(size_t i=0; i<relations.size(); i++)
    {
        // sort ascending, according to remaining primes (bigger than currentPrime)
        comparator.minPrime = currentPrime;
        sort(relations.begin()+i, relations.end(), comparator);

        auto nextPrimeIterator = upper_bound(relations[i].oddPrimePowers.indices.begin(), relations[i].oddPrimePowers.indices.end(), currentPrime);

        // if no more primes
        if(nextPrimeIterator == relations[i].oddPrimePowers.indices.end())
            break;

        // go to next prime
        currentPrime = *nextPrimeIterator;

        relations[i].dependsOnPrime = currentPrime;

        // remove all other primes in current relations bigger than currentPrime
        for(auto primeIt = nextPrimeIterator+1; primeIt != relations[i].oddPrimePowers.indices.end(); ++primeIt)
        {
            uint32_t primeToRemove = *primeIt;
            uint32_t k;
            for(k=i+1; k<relations.size(); k++)
            {
                auto start = relations[k].oddPrimePowers.indices.begin();
                auto last = relations[k].oddPrimePowers.indices.end();
                if(find(start, last, currentPrime) == last)
                    break;


                //cout << "\t eliminating " << primeToRemove << " at: ", relations[k].print();

                auto lowerBoundIt = lower_bound(start, last, primeToRemove);
                if(lowerBoundIt == last || *lowerBoundIt > primeToRemove)
                {
                    relations[k].oddPrimePowers.indices.insert(lowerBoundIt, primeToRemove);
                }
                else
                {
                    relations[k].oddPrimePowers.indices.erase(lowerBoundIt);
                }

                //cout << "\t afterwards: ", relations[k].print();
            }

            //assert that no other has 1 at front
            for(k=k+1; k<relations.size(); k++)
            {
                auto start = relations[k].oddPrimePowers.indices.begin();
                auto last = relations[k].oddPrimePowers.indices.end();
                if(find(start, last, currentPrime) != last)
                    throw logic_error("asd");
            }

        }

        relations[i].oddPrimePowers.indices.erase(nextPrimeIterator+1, relations[i].oddPrimePowers.indices.end());

    }
}


PrimeFactorization QuadraticSieve::factorizeOverBase(const BigInt& number) const
{
    PrimeFactorization result;
    BigInt x = number;
    BigInt q, r;
    for(size_t i=0; i<factorBase.size() && x > 1; i++)
    {
        smallPrime_t prime = factorBase[i];
        uint32_t power = 0;
        do
        {
            mpz_fdiv_qr_ui(q.get_mpz_t(), r.get_mpz_t(), x.get_mpz_t(), prime);            
            if(r == 0)
            {
                power++;
                x = q;
            }

        }while(r == 0);
        if(power > 0)
        {
            result.add(prime, power);
        }
    }
    if(x == 1)
    {
        return result;
    }
    return PrimeFactorization();
}


double binarySolve(function<double(double)> f, double y)
{
    double xLo = 2;
    double xHi = 2;
    while(f(xLo) > y)
        xLo /= 2;
    while(f(xHi) < y)
        xHi *= 2;
    do
    {
        double avg = (xHi + xLo) / 2;
        if(f(avg) > y)
            xHi = avg;
        else
            xLo = avg;
        //cout << "(" << xLo << ", " << xHi << ") " <<  f((xHi+xLo)/2) <<endl;
    }while((xHi-xLo) / ((xHi + xLo) / 2) > 1e-10);

    return (xHi + xLo) / 2;
}


void QuadraticSieve::createFactorBase(size_t numberOfPrimes) 
{
    uint32_t numbersToCheck = (uint32_t) binarySolve(
                                            [](double x){return x / log(x);}, 
                                            numberOfPrimes) * 2;

    vector<uint32_t> isPrime(numbersToCheck, true);

    vector<smallPrime_t> primes;
    primes.reserve(numberOfPrimes);

    for(uint32_t i = 2; i<numbersToCheck && primes.size() < numberOfPrimes; i++)
    {
        if(!isPrime[i])
            continue;

        primes.push_back(i);

        for(uint32_t k=i; k<numbersToCheck; k+=i)
            isPrime[k] = false;
    }

    factorBase = move(primes);
}

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

PrimeFactorization sqrt(const PrimeFactorization& self)
{
    return self.sqrt();
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

void PrimeFactorization::add(const smallPrime_t& prime, uint32_t power)
{
    primePowers.push_back({prime, power});
}

bool Relation::isPerfectCongruence() const
{
    return oddPrimePowers.empty();
}