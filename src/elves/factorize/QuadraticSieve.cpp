#include "QuadraticSieve.h"
#include <algorithm>
#include <future>
#include <cassert>

using namespace std;

const pair<BigInt,BigInt> QuadraticSieve::TRIVIAL_FACTORS(0,0);

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

    cout << "found " << relations.size() << " relations" << endl;

    // bring relations into lower diagonal form
    cout << "performing gaussian elimination ..." << endl;
    performGaussianElimination();

    cout << "combining random congruences ..." << endl;

    return searchForRandomCongruence(100);
}

pair<BigInt,BigInt> QuadraticSieve::factorsFromCongruence(const BigInt& a, const BigInt& b) const
{
    BigInt p = gcd(a+b, n);
    BigInt q = gcd(absdiff(a,b), n);
    return make_pair(p, q);
}

bool QuadraticSieve::isNonTrivial(const pair<BigInt,BigInt>& factors) const
{
    const BigInt& p = factors.first;
    const BigInt& q = factors.second;

    return p>1 && p<n && q>1 && q<n;
}


pair<BigInt, BigInt> QuadraticSieve::sieve()
{
    BigInt intervalSize = exp(sqrt(log(n)*log(log(n))));
    BigInt intervalStart = sqrt(n) + 1;
    BigInt intervalEnd = sqrt(n)+ 1 + intervalSize;
    intervalEnd = (sqrt(2*n) < intervalEnd) ? sqrt(2*n) : intervalEnd;

    cout << "sieving interval: " << (intervalEnd - intervalStart) << endl;
    return sieveIntervalFast(intervalStart, intervalEnd, factorBase.size() + 2);
}

// returns a list of numbers, which quadratic residues are (probable) smooth over the factor base
vector<BigInt> QuadraticSieve::sieveSmoothSquares(const BigInt& start, const BigInt& end) const
{
    BigInt intervalLength = (end-start);
    size_t blockSize = intervalLength.get_ui();
    vector<uint32_t> logs(blockSize+1);
    BigInt x, remainder;
    uint32_t logTreshold = (int)(lb(n));

    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % n;
        logs[i] = log_2_22(remainder);
    }

    // now with prime powers
    cout << "starting with logarithmic sieving ..." << endl;
    for(const smallPrime_t& smallPrime : factorBase)
    {
        BigInt prime(smallPrime);
        uint32_t primeLog = log_2_22(prime);
        uint32_t i = 1;
        BigInt primePower = prime;
        for(; primePower < n; i++, primePower*=prime)
        {
            vector<BigInt> roots = squareRootsModPrimePower(n%primePower, prime, i);
            for(const BigInt& root : roots)
            {
                BigInt offset = (primePower + root - (start % primePower)) % primePower;
                for(BigInt j=offset; j<=blockSize; j+=primePower)
                {
                    logs[j.get_ui()] -= primeLog;
                }
            }
        }
    }

    //second scan for smooth numbers
    BigInt biggestPrime(factorBase.back());


    vector<BigInt> result;

    for(uint32_t i=0; i<=blockSize; i++)
    {
        if(logs[i] < logTreshold) // probable smooth
        {
            result.emplace_back(start+i);
        }
    }

    return result;
}

pair<BigInt, BigInt> QuadraticSieve::sieveIntervalFast(const BigInt& start, const BigInt& end, size_t maxRelations)
{
    vector<BigInt> smooths = sieveSmoothSquares(start, end);

    for(const BigInt& x : smooths)
    {
        BigInt remainder = (x*x) % n;

        PrimeFactorization factorization = factorizeOverBase(remainder);
        if(factorization.empty())
        {
            cerr << "false alarm !!! (should not happend)" << endl;
            continue;
        }

        Relation relation(x, factorization);

        if(relation.isPerfectCongruence())
        {
            auto factors = factorsFromCongruence(x, sqrt(factorization).multiply());
            if(isNonTrivial(factors))
            {
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



struct RelationComparator {
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
};


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

// a is not a quadratic residue
bool hasRootModPrime(const BigInt& a, const BigInt& prime)
{
    BigInt remainder = a % prime;
    int jacobi = mpz_jacobi(remainder.get_mpz_t(), prime.get_mpz_t());
    return jacobi != -1;
}

vector<BigInt> QuadraticSieve::squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power)
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

vector<BigInt> QuadraticSieve::liftRoots(const vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower)
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


BigInt QuadraticSieve::liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power)
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

BigInt QuadraticSieve::rootModPrime(const BigInt& a, const BigInt& p)
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
