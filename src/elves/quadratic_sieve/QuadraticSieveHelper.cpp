#include "QuadraticSieveHelper.h"
#include "common/Utils.h"
#include "PrimeFactorization.h"
#include <algorithm>
#include <cassert>
#include <iostream>

// Uncomment this to skip the (non-parallel) gaussian elimination stage,
// which takes very long for larger input numbers
//#define SKIP_LINEAR_ALGEBRA

using namespace std;

const BigInt MAX_INTERVAL_SIZE(BigInt(1073741824) * 4);
const BigInt USE_PARALLEL_FACTOR_STAGE_THRESHOLD("10000000000000000000000");
const size_t ADDITIONAL_RELATIONS = 2;
const size_t SEARCH_CONGRUENCES_ITERATIONS = 100;

pair<BigInt, BigInt> factorSmoothSquaresSequential(
    const size_t maxRelations,
    const SmoothSquareList& smooths,
    const BigInt& number,
    vector<Relation>& relations,
    const FactorBase& factorBase
)
{
    for (const BigInt& x : smooths) 
    {
        BigInt remainder = (x*x) % number;

        PrimeFactorization factorization = QuadraticSieveHelper::factorizeOverBase(remainder, factorBase);
        if(factorization.empty())
        {
            cerr << "false alarm !!! (should not happen)" << endl;
            continue;
        }

        Relation relation(x, factorization);
        if(relation.isPerfectCongruence())
        {
            auto factors = QuadraticSieveHelper::factorsFromCongruence(x, sqrt(factorization).multiply(), number);
            if(QuadraticSieveHelper::isNonTrivial(factors, number))
                return factors;
        }
        relations.push_back(relation);

        if(relations.size() >= maxRelations)
            break;
    }

    return QuadraticSieveHelper::TRIVIAL_FACTORS;
}

pair<BigInt, BigInt> factorSmoothSquaresParallel(
    const size_t maxRelations,
    const SmoothSquareList& smooths,
    const BigInt& number,
    vector<Relation>& relations,
    const FactorBase& factorBase
)
{
    cout << "Factoring smooth squares (in parallel) ..." << endl;
    volatile bool loopFinished = false;
    pair<BigInt, BigInt> earlyResult(QuadraticSieveHelper::TRIVIAL_FACTORS);

    #pragma omp parallel for
    for (size_t i=0; i<maxRelations; ++i)
    {
        const BigInt& x = smooths[i];

        #pragma omp flush(loopFinished)
        if (loopFinished) continue;

        BigInt remainder = (x*x) % number;

        PrimeFactorization factorization = QuadraticSieveHelper::factorizeOverBase(remainder, factorBase);
        if(factorization.empty())
        {
            cerr << "false alarm !!! (should not happen)" << endl;
            continue;
        }

        Relation relation(x, factorization);

        if(relation.isPerfectCongruence())
        {
            auto factors = QuadraticSieveHelper::factorsFromCongruence(x, sqrt(factorization).multiply(), number);
            if(QuadraticSieveHelper::isNonTrivial(factors, number))
            {
                #pragma omp critical
                {
                    earlyResult = factors;
                    loopFinished = true;
                }
            }
        }

        #pragma omp critical
        {
            relations.push_back(relation);

            if(relations.size() >= maxRelations)
                loopFinished = true;
        }
    }

    sort(relations.begin(), relations.end(), [](const Relation& a, const Relation& b) {
        return a.a < b.a;
    });

    return earlyResult;
}

pair<BigInt, BigInt> sieveIntervalFast(
    const BigInt& start,
    const BigInt& end,
    vector<Relation>& relations,
    const FactorBase& factorBase,
    const BigInt& number,
    SieveSmoothSquaresCallback sieveSmoothSquaresCallback
)
{
    const size_t maxRelations = factorBase.size() + ADDITIONAL_RELATIONS;
    SmoothSquareList smooths = sieveSmoothSquaresCallback(start, end, number, factorBase);
    
    auto factorFunction = &factorSmoothSquaresSequential;

    if (number >= USE_PARALLEL_FACTOR_STAGE_THRESHOLD)
        factorFunction = &factorSmoothSquaresParallel;

    return factorFunction(
        maxRelations, 
        smooths, 
        number, 
        relations, 
        factorBase
    );
}

BigInt guessIntervalSize(const BigInt& number)
{
    return exp(sqrt(log(number)*log(log(number))));
}

pair<BigInt, BigInt> sieve(
    vector<Relation>& relations,
    const FactorBase& factorBase,
    const BigInt& number,
    SieveSmoothSquaresCallback sieveSmoothSquaresCallback
)
{
    BigInt intervalSize = min(guessIntervalSize(number), MAX_INTERVAL_SIZE);
    BigInt intervalStart = sqrt(number) + 1;
    BigInt intervalEnd = sqrt(number)+ 1 + intervalSize;
    intervalEnd = (sqrt(2*number) < intervalEnd) ? sqrt(2*number) : intervalEnd;

    cout << "sieving interval: " << (intervalEnd - intervalStart) << endl;
    return sieveIntervalFast(
        intervalStart,
        intervalEnd,
        relations,
        factorBase,
        number,
        sieveSmoothSquaresCallback
    );
}

namespace QuadraticSieveHelper
{

const pair<BigInt,BigInt> TRIVIAL_FACTORS(0,0);

int guessFactorBaseSize(const BigInt& number)
{
    return (int)exp(0.5*sqrt(log(number)*log(log(number))));
}

pair<BigInt, BigInt> factor(const BigInt& number, SieveSmoothSquaresCallback sieveCallback)
{
    int factorBaseSize = guessFactorBaseSize(number);
    cout << "factorBaseSize " << factorBaseSize << endl;
    auto factorBase = createFactorBase(factorBaseSize);

    // sieve
    cout << "sieving relations ..." << endl;
    vector<Relation> relations;
    pair<BigInt, BigInt> factors = sieve(relations, factorBase, number, sieveCallback);
    if(isNonTrivial(factors, number))
        return factors;

    cout << "found " << relations.size() << " relations" << endl;
#ifdef SKIP_LINEAR_ALGEBRA
    return TRIVIAL_FACTORS;
#endif

    // bring relations into lower diagonal form
    cout << "performing gaussian elimination ..." << endl;
    performGaussianElimination(relations);
    cout << "combining random congruences ..." << endl;
    return searchForRandomCongruence(factorBase, number, SEARCH_CONGRUENCES_ITERATIONS, relations);
}

pair<BigInt,BigInt> factorsFromCongruence(const BigInt& a, const BigInt& b, const BigInt& number)
{
    BigInt p = gcd(a+b, number);
    BigInt q = gcd(absdiff(a,b), number);
    return make_pair(p, q);
}

bool isNonTrivial(const pair<BigInt,BigInt>& factors, const BigInt& number)
{
    const BigInt& p = factors.first;
    const BigInt& q = factors.second;

    return p>1 && p<number && q>1 && q<number;
}

static pair<BigInt,BigInt> pickRandomCongruence(const FactorBase& factorBase, const BigInt& number, function<int()> randomGenerator, const Relations& relations)
{
    vector<bool> primeMask(factorBase.back()+1);

    vector<Relation> selectedRelations;
    for(auto relIter = relations.rbegin(); relIter != relations.rend(); relIter++)
    {
        const Relation& relation = *relIter;

        if((relation.dependsOnPrime == 0 && randomGenerator() == 1) || (relation.dependsOnPrime > 0 && primeMask[relation.dependsOnPrime]))
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
        a = (a * relation.a) % number;
        factorization = factorization.combine(relation.primeFactorization);
    }

    BigInt b = factorization.sqrt().multiply();

    return factorsFromCongruence(a,b, number);
}

pair<BigInt,BigInt> searchForRandomCongruence(const FactorBase& factorBase, const BigInt& number, size_t times, const Relations& relations)
{
    uniform_int_distribution<int> binaryDistribution(0, 1);
    mt19937 randomEngine;
    auto randomGenerator = bind(binaryDistribution, randomEngine);

    for(size_t i=0; i<times; i++)
    {
        pair<BigInt, BigInt> result = pickRandomCongruence(factorBase, number, randomGenerator, relations);
        if(isNonTrivial(result, number))
            return result;
    }
    return TRIVIAL_FACTORS;
}

class RelationComparator 
{
public:
    RelationComparator(smallPrime_t minPrime) :
        minPrime(minPrime)
    {}

    bool operator() (const Relation& a, const Relation& b)
    {
        auto aStart = upper_bound(a.oddPrimePowers.indices.begin(), a.oddPrimePowers.indices.end(), minPrime);
        auto bStart = upper_bound(b.oddPrimePowers.indices.begin(), b.oddPrimePowers.indices.end(), minPrime);

        if(aStart < a.oddPrimePowers.indices.end() && bStart < b.oddPrimePowers.indices.end())
        {
            if(*aStart < *bStart)
                return true;
            if (*aStart > *bStart)
                return false;
            return (a.oddPrimePowers.indices.end() - aStart) < (b.oddPrimePowers.indices.end() - bStart);
        }

        if(aStart < a.oddPrimePowers.indices.end())
            return true;
        else
            return false;
    }

private:
    smallPrime_t minPrime;
};


void performGaussianElimination(Relations& relations)
{
    uint32_t currentPrime = 0;

    for(size_t i=0; i<relations.size(); i++)
    {
        // swap next smallest relation to top, according to remaining primes (bigger than currentPrime)
        RelationComparator comparator(currentPrime);

        auto minRelation = min_element(relations.begin()+i, relations.end(), comparator);
        swap(*minRelation, relations[i]);

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
                if (!binary_search(start, last, currentPrime))
                    continue;

                auto lowerBoundIt = lower_bound(start, last, primeToRemove);
                if(lowerBoundIt == last || *lowerBoundIt > primeToRemove)
                {
                    relations[k].oddPrimePowers.indices.insert(lowerBoundIt, primeToRemove);
                }
                else
                {
                    relations[k].oddPrimePowers.indices.erase(lowerBoundIt);
                }
            }

            //assert that no other has 1 at front
            assert([&]()
               {
                    for(k=k+1; k<relations.size(); k++)
                    {
                        auto start = relations[k].oddPrimePowers.indices.begin();
                        auto last = relations[k].oddPrimePowers.indices.end();
                        if (find(start, last, currentPrime) != last)
                            return false;
                    }
                    return true;
               }()
            );
        }

        relations[i].oddPrimePowers.indices.erase(nextPrimeIterator+1, relations[i].oddPrimePowers.indices.end());

    }
}


PrimeFactorization factorizeOverBase(const BigInt& number, const FactorBase& factorBase)
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

// a is not a quadratic residue
bool hasRootModPrime(const BigInt& a, const BigInt& prime)
{
    BigInt remainder = a % prime;
    int jacobi = mpz_jacobi(remainder.get_mpz_t(), prime.get_mpz_t());
    return jacobi != -1;
}

vector<BigInt> squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power)
{
    vector<BigInt> roots;

    if (!hasRootModPrime(a, prime))
        return roots;

    BigInt basicRoot = rootModPrime(a, prime);
    roots.push_back(basicRoot);
    if((prime - basicRoot) % prime != basicRoot)
        roots.push_back(prime - basicRoot);

    for(uint32_t i=2; i<=power; i++)
    {
        roots = liftRoots(roots, a, prime, i);
    }

    return roots;
}

vector<BigInt> liftRoots(const vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower)
{
    vector<BigInt> newRoots;
    BigInt currentPrimePower, nextPrimePower;
    mpz_pow_ui(currentPrimePower.get_mpz_t(), prime.get_mpz_t(), nextPower-1);
    mpz_pow_ui(nextPrimePower.get_mpz_t(), prime.get_mpz_t(), nextPower);

    for(const BigInt& root : roots)
    {
        BigInt b = (2*root) % prime;
        if(b == 0)
        {
            if((root*root - a) % nextPrimePower == 0)
            {
                for(BigInt i=0; i<prime; ++i)
                {
                    newRoots.emplace_back(root + i*currentPrimePower);
                }
            }
        }
        else
        {
            BigInt inverseB;
            mpz_invert(inverseB.get_mpz_t(), b.get_mpz_t(), prime.get_mpz_t());
            BigInt c = ((root*root - a) * inverseB) % nextPrimePower;
            newRoots.emplace_back((nextPrimePower + root - c) % nextPrimePower);
        }
    }
    return newRoots;
}


BigInt liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power)
{
    BigInt pi = p;
    BigInt x = root;
    BigInt b, inverseB;
    for(uint32_t i=2; i<power; i++)
    {
        b = x * 2;
        mpz_invert(inverseB.get_mpz_t(), b.get_mpz_t(), p.get_mpz_t());
        pi *= p;
        x = (x - (inverseB*(((x*x)%pi) - a)%pi)) % pi;
    }
    return x;
}

BigInt rootModPrime(const BigInt& a, const BigInt& p)
{
    if(a >= p)
        return rootModPrime(a % p, p);

    if(!hasRootModPrime(a, p))
        throw logic_error("Unable to take root of quadratic non-residue.");

    if(p == 2)
    {
        return a;
    }

    // check simplest cases: p = 3, 5, 7 mod 8
    BigInt pRemEight = p % 8;
    if(pRemEight == 3 || pRemEight == 7)
    {
        BigInt power = (p+1)/4;
        BigInt x;
        mpz_powm(x.get_mpz_t(), a.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
        return x;
    }
    if(pRemEight == 5)
    {
        BigInt power = (p+3)/8;
        BigInt x;
        mpz_powm(x.get_mpz_t(), a.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
        BigInt c;
        mpz_powm_ui(c.get_mpz_t(), x.get_mpz_t(), 2, p.get_mpz_t());
        if(c != a)
        {
            BigInt scale(2);
            power = (p-1)/4;
            mpz_powm(scale.get_mpz_t(), scale.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
            x = (x * scale) % p;

        }
        return x;
    }

    gmp_randstate_t rstate;
    gmp_randinit_mt(rstate);
    BigInt d;
    int jacobi;
    do{
        mpz_urandomm(d.get_mpz_t(), rstate, p.get_mpz_t());
        jacobi = mpz_jacobi(d.get_mpz_t(), p.get_mpz_t());
    }while(!(d>1 && jacobi == -1));
    gmp_randclear(rstate);

    BigInt t;
    BigInt pMinusOne(p-1);
    BigInt two(2);
    BigInt one(1);
    mp_bitcnt_t s = mpz_remove(t.get_mpz_t(), pMinusOne.get_mpz_t(), two.get_mpz_t());

    BigInt A, D, m, x, Dm, ADm, tmp, power;
    mpz_powm(A.get_mpz_t(), a.get_mpz_t(), t.get_mpz_t(), p.get_mpz_t());
    mpz_powm(D.get_mpz_t(), d.get_mpz_t(), t.get_mpz_t(), p.get_mpz_t());
    m = 0;

    for(uint i=1; i<s; i++)
    {
        mpz_ui_pow_ui(power.get_mpz_t(), 2, s - 1 - i);
        mpz_powm(Dm.get_mpz_t(), D.get_mpz_t(), m.get_mpz_t(), p.get_mpz_t());
        ADm = A * Dm;
        mpz_powm(tmp.get_mpz_t(), ADm.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
        if(tmp+1==p)
        {
            m += one << i;
        }
    }
    power = (t+1) / 2;
    mpz_powm(A.get_mpz_t(), a.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
    power = m / 2;
    mpz_powm(D.get_mpz_t(), D.get_mpz_t(), power.get_mpz_t(), p.get_mpz_t());
    x = (A*D) % p;
    return x;
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
    } while((xHi-xLo) / ((xHi + xLo) / 2) > 1e-10);

    return (xHi + xLo) / 2;
}


std::vector<smallPrime_t> createFactorBase(size_t numberOfPrimes)
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

        for(uint32_t k=i*i; k<numbersToCheck; k+=i)
            isPrime[k] = false;
    }

    return primes;
}

}
