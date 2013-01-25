#include "QuadraticSieve.h"
#include <algorithm>
#include <future>
#include "elves/factorize/cuda/Factorize.h"
#include "elves/factorize/cuda/NumberHelper.h"
#include <cuda-utils/Memory.h>

using namespace std;

const pair<BigInt,BigInt> QuadraticSieve::TRIVIAL_FACTORS(0,0);

extern void sieveIntervalWrapper(PNumData pn, uint32_t* logs, uint32_t* rootsModPrime, uint32_t* factorBase, int factorBaseSize, PNumData pStart, PNumData pEnd);

pair<BigInt, BigInt> QuadraticSieve::factorize()
{
    int factorBaseSize = (int)exp(0.5*sqrt(log(n)*log(log(n))));
    cout << "factorBaseSize" << factorBaseSize << endl;
    createFactorBase(factorBaseSize);

    // sieve
    cout << "sieving relations ..." << endl;
    pair<BigInt, BigInt> factors = sieve();
    if(isNonTrivial(factors))
        return factors;

    cout << "found " << relations.size() << " realtions" << endl;

    // bring relations into lower diagonal form
    cout << "performing gaussian elimination ..." << endl;
    performGaussianElimination();

    cout << "combining random congruences ..." << endl;

    return searchForRandomCongruence(100);
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
    uint64_t intervalSize = (uint64_t)exp(sqrt(log(n)*log(log(n))));
    BigInt intervalStart = sqrt(n) + 1;
    BigInt intervalEnd = sqrt(n)+ 1 + intervalSize;
    return sieveIntervalFast(intervalStart, intervalEnd, factorBase.size() + 2);
    //return sieveIntervalCuda(intervalStart, intervalEnd, factorBase.size() + 2);
}


pair<BigInt, BigInt> QuadraticSieve::sieveIntervalCuda(const BigInt& start, const BigInt& end, size_t maxRelations)
{
    BigInt intervalLength = (end-start);
    size_t blockSize = intervalLength.get_ui();
    cout << "sieving interval: " << blockSize << endl;

    vector<uint32_t> logs(blockSize+1);

    BigInt x, remainder;
    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % n;
        logs[i] = log_2_22(remainder);
    }

    vector<BigInt> rootsModPrime(factorBase.size());
    for (auto primeIter = factorBase.begin(); primeIter != factorBase.end(); ++primeIter)
    {
        BigInt root = rootModPrime(n, *primeIter);
        rootsModPrime.push_back(root);
    }
    
    CudaUtils::Memory<uint32_t> logs_d(blockSize+1);
    CudaUtils::Memory<uint32_t> rootsModPrime_d = NumberHelper::BigIntsToNumbers(rootsModPrime); 
    CudaUtils::Memory<uint32_t> factorBase_d(factorBase.size());
    CudaUtils::Memory<uint32_t> start_d = NumberHelper::BigIntToNumber(start);
    CudaUtils::Memory<uint32_t> end_d = NumberHelper::BigIntToNumber(end);
    CudaUtils::Memory<uint32_t> n_d = NumberHelper::BigIntToNumber(n);
    
    factorBase_d.transferFrom(&factorBase[0]);
    logs_d.transferFrom(&logs[0]);

    sieveIntervalWrapper(n_d.get(), logs_d.get(), rootsModPrime_d.get(), factorBase_d.get(), factorBase.size(), start_d.get(), end_d.get());

    vector<uint32_t> gpulogs = (NumberHelper::NumbersToUis(logs_d));
    
    swap(gpulogs, logs);

    //second scan for smooth numbers
    BigInt biggestPrime(factorBase.back());
    //uint32_t logTreshold = (int)(log_2_22(biggestPrime) + lb(n));
    uint32_t logTreshold = (int)(lb(n));
    for(uint32_t i=0; i<=blockSize; i++)
    {
        //cout << logs[i] << " < " << logTreshold << endl;
        if(logs[i] < logTreshold) // probable smooth
        {
            x = start + i;
            remainder = (x*x) % n;

            PrimeFactorization factorization = factorizeOverBase(remainder);
            if(factorization.empty())
            {
                cerr << "false alarm !!! (should not happend)" << endl;
                continue;
            }


            Relation relation(x, factorization);
            //cout << "NEW: ", print(relation);
            //cout << "R#=" << relations.size() << endl;

            uint32_t logSum = 0;
            for(auto pp : factorization.oddPrimePowers().indices)
            {
                BigInt bigpp(pp);
                logSum += log_2_22(bigpp);
            }
            //factorization.print();
            /*cout << "ln(x)=" << log_2_22(remainder) 
                << ", sum(ln)=" << logSum 
                << ", log[i]=" << logs[i] << endl;*/

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

            if(relations.size() >= maxRelations)
                break;

        }
    }

    return TRIVIAL_FACTORS;
}

pair<BigInt, BigInt> QuadraticSieve::sieveIntervalFast(const BigInt& start, const BigInt& end, size_t maxRelations)
{
    BigInt intervalLength = (end-start);
    size_t blockSize = intervalLength.get_ui();
    cout << "sieving interval: " << blockSize << endl;

    vector<uint32_t> logs(blockSize+1);

    BigInt x, remainder;
    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % n;
        logs[i] = log_2_22(remainder);
    }

    // no prime powers
    for(const smallPrime_t& prime : factorBase)
    {
        BigInt root = rootModPrime(n, prime);
        if(root == 0)
            continue;

        for(int z = 0; z<2; z++)
        {

            BigInt offset = (prime + root - (start % prime)) % prime;

            //cout << root << "^2-" << n << " is dividable by " << prime << endl; 
            //cout << (start+offset) << "^2-" << n << " [offset=" << offset << "] should be dividable by " << prime << endl; 
            //cout << "start:" << start << endl;
            //cout << "offset: " << offset << endl;
            BigInt bigPrime(prime);
            uint32_t primeLog = log_2_22(bigPrime);
            for(uint32_t i=offset.get_ui(); i<=blockSize; i+=prime)
            {
                logs[i] -= primeLog;

                x = start + i;
                if(((x*x)%n)%bigPrime != 0)
                {
                    cout << "(x*x)%n=" << ((x*x)%n) << " is not dividiable by " << prime << endl;
                }
                //if(i == 5690)
                //    cout << "(x*x)%n=" << ((x*x)%n) << " is dividiable by " << prime << endl;
                
            }

            if(prime-root == root)
                break;
            else
                root = prime - root;
        }
    }

    //second scan for smooth numbers
    BigInt biggestPrime(factorBase.back());
    //uint32_t logTreshold = (int)(log_2_22(biggestPrime) + lb(n));
    uint32_t logTreshold = (int)(lb(n));
    for(uint32_t i=0; i<=blockSize; i++)
    {
        //cout << logs[i] << " < " << logTreshold << endl;
        if(logs[i] < logTreshold) // probable smooth
        {
            x = start + i;
            remainder = (x*x) % n;

            PrimeFactorization factorization = factorizeOverBase(remainder);
            if(factorization.empty())
            {
                cerr << "false alarm !!! (should not happend)" << endl;
                continue;
            }


            Relation relation(x, factorization);
            //cout << "NEW: ", print(relation);
            //cout << "R#=" << relations.size() << endl;

            uint32_t logSum = 0;
            for(auto pp : factorization.oddPrimePowers().indices)
            {
                BigInt bigpp(pp);
                logSum += log_2_22(bigpp);
            }
            //factorization.print();
            /*cout << "ln(x)=" << log_2_22(remainder) 
                << ", sum(ln)=" << logSum 
                << ", log[i]=" << logs[i] << endl;*/

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

            if(relations.size() >= maxRelations)
                break;

        }
    }

    return TRIVIAL_FACTORS;
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
        cout << "R#=" << relations.size() << endl;

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

        if(relations.size() >= maxRelations)
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

    //cout << " depends on: " << r.dependsOnPrime;

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
    uint32_t minPrime;
} RelationComparator;


void QuadraticSieve::performGaussianElimination()
{
    RelationComparator comparator;
    uint32_t currentPrime = 0;

    for(size_t i=0; i<relations.size(); i++)
    {
        // swap next smallest relation to top, according to remaining primes (bigger than currentPrime)
        comparator.minPrime = currentPrime;

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
                if(find(start, last, currentPrime) == last)
                    continue;


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


BigInt QuadraticSieve::rootModPrime(const BigInt& a, const BigInt& p)
{
    if(a > p)
        return rootModPrime(a % p, p);

    int jacobi = mpz_jacobi(a.get_mpz_t(), p.get_mpz_t());

    if(jacobi != 1) // a is not a quadratic residue
        return 0;

    
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
    do{
        mpz_urandomm(d.get_mpz_t(), rstate, p.get_mpz_t());
        jacobi = mpz_jacobi(d.get_mpz_t(), p.get_mpz_t());
    }while(!(d>1 && jacobi == -1));

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





    /*
    BigInt rem;
    for(BigInt i=1; i<primeMod; i++)
    {
        mpz_powm_ui(rem.get_mpz_t(), i.get_mpz_t(), 2, primeMod.get_mpz_t());
        if(rem == n)
        {
            return i;
        }
    }
    return 0;
    */
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
