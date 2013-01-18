#include "factorize/BigInt.h"

#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <functional>
#include <utility>
#include <bitset>
#include <map>
#include <memory>
#include <algorithm>
#include <random>

using namespace std;
using namespace chrono;
using namespace testing;


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

vector<uint32_t> createFirstNPrimes(uint32_t n)
{
    uint32_t neededNumbers = (uint32_t) binarySolve(
                                            [](double x){return x / log(x);}, 
                                            n) + 1;

    vector<uint32_t> isPrime(neededNumbers, true);

    vector<uint32_t> primes;
    primes.reserve(n);

    for(uint32_t i = 2; i<neededNumbers && primes.size() < n; i++)
    {
        //cout << "inspecting: " << i << endl;
        if(!isPrime[i])
            continue;

        //cout << "prime: " << i << endl;

        primes.push_back(i);

        for(uint32_t k=i; k<neededNumbers; k+=i)
            isPrime[k] = false;
    }

    return primes;
}




const size_t numberOfPrimes = 50;


bool getPrimeFactors(BigInt& n, vector<uint32_t> primes, vector<uint32_t>& factors)
{
    BigInt x = n;
    BigInt q, r;
    for(size_t i=0; i<primes.size() && x > 1; i++)
    {
        uint32_t prime = primes[i];

        do
        {
            mpz_fdiv_qr_ui(q.get_mpz_t(), r.get_mpz_t(), x.get_mpz_t(), prime);            
            if(r == 0)
            {
                factors[i]++;
                x = q;
            }

        }while(r == 0);
    }
    if(x > 1)
    {
        return false;
    }
    return true;
}


class Relation
{
public:
    BigInt a;
    vector<uint32_t> primeFactorization;
    vector<uint32_t> oddPrimeFactors;
    uint32_t dependsOnPrime;

    Relation() : a(0), primeFactorization(0), oddPrimeFactors(0), dependsOnPrime(0)
    {}

    void print() const 
    {        
        cout << a << "^2 =\t";
        cout << ((find(oddPrimeFactors.begin(), oddPrimeFactors.end(), 2) == oddPrimeFactors.end()) ? 0 : 1) << " ";
        cout << ((find(oddPrimeFactors.begin(), oddPrimeFactors.end(), 3) == oddPrimeFactors.end()) ? 0 : 1) << " ";
        cout << ((find(oddPrimeFactors.begin(), oddPrimeFactors.end(), 5) == oddPrimeFactors.end()) ? 0 : 1) << " ";
        cout << ((find(oddPrimeFactors.begin(), oddPrimeFactors.end(), 7) == oddPrimeFactors.end()) ? 0 : 1) << " ";
        cout << "(";
        for(uint32_t p : oddPrimeFactors)
        {
            cout << p << " ";
        }
        cout << ")";

        cout << " [";
        for(uint32_t p : primeFactorization)
        {
            cout << p << " ";
        }
        cout << "]";

        cout << endl;
    }
};

typedef struct {
    bool operator() (const Relation& a, const Relation& b) 
    { 
        auto aStart = upper_bound(a.oddPrimeFactors.begin(), a.oddPrimeFactors.end(), minPrime);
        auto bStart = upper_bound(b.oddPrimeFactors.begin(), b.oddPrimeFactors.end(), minPrime);

        auto aIt=aStart, bIt=bStart;
        for(aIt=aStart, bIt=bStart; aIt != a.oddPrimeFactors.end() && bIt != b.oddPrimeFactors.end(); aIt++, bIt++)
        {
            if(*aIt < *bIt)
                return true;
            else if (*aIt > *bIt)
                return false;
        }
        //return false;
        return aIt != a.oddPrimeFactors.end();
    }
/*
    bool operator() (const Relation& a, const Relation& b) 
    { 
        cout << "a = ", a.print();
        cout << "b = ", b.print();

        cout << "detect upper bounds" << endl;
        auto aStart = upper_bound(a.oddPrimeFactors.begin(), a.oddPrimeFactors.end(), minPrime);
        auto bStart = upper_bound(b.oddPrimeFactors.begin(), b.oddPrimeFactors.end(), minPrime);

        cout << "check 1" << endl;
        if(bStart == b.oddPrimeFactors.end())
            return true;
        cout << "check 2" << endl;
        if(aStart == a.oddPrimeFactors.end())
            return false;

        cout << "loopy 2" << endl;
        auto aIt=aStart, bIt=bStart;
        for(aIt=aStart, bIt=bStart; aIt != a.oddPrimeFactors.end() && bIt != b.oddPrimeFactors.end(); aIt++, bIt++)
        {
            if(*aIt < *bIt)
                return true;
            else if (*aIt > *bIt)
                return false;

            //break;
        }
        return false;
        cout << "last length comparison" << endl;
        return (a.oddPrimeFactors.end() - aIt) < (b.oddPrimeFactors.end() - bIt);
    }*/
    uint32_t minPrime;
} RelationComparator;


void gaussEliminiation(vector<Relation>& relations)
{
    RelationComparator comparator;
    uint32_t currentPrime = 0;
    uint32_t i;



    for(i=0; i<relations.size(); i++)
    {
        // sort ascending, according to remaining primes (bigger than currentPrime)
        comparator.minPrime = currentPrime;
        sort(relations.begin()+i, relations.end(), comparator);

/*
        cout << "---sorted--- with primes > " << currentPrime << endl;
        for(auto r : relations)
        {
            r.print();
        }
        cout << "------------" << endl;

        cout << "looking at: ", relations[i].print();
 */
        auto nextPrimeIterator = upper_bound(relations[i].oddPrimeFactors.begin(), relations[i].oddPrimeFactors.end(), currentPrime);

        // if no more primes
        if(nextPrimeIterator == relations[i].oddPrimeFactors.end())
            break;

        // go to next prime
        currentPrime = *nextPrimeIterator;

        relations[i].dependsOnPrime = currentPrime;

        // remove all other primes in current relations bigger than currentPrime
        for(auto primeIt = nextPrimeIterator+1; primeIt != relations[i].oddPrimeFactors.end(); ++primeIt)
        {
            uint32_t primeToRemove = *primeIt;
            uint32_t k;
            for(k=i+1; k<relations.size(); k++)
            {
                auto start = relations[k].oddPrimeFactors.begin();
                auto last = relations[k].oddPrimeFactors.end();
                if(find(start, last, currentPrime) == last)
                    break;


                //cout << "\t eliminating " << primeToRemove << " at: ", relations[k].print();

                auto lowerBoundIt = lower_bound(start, last, primeToRemove);
                if(lowerBoundIt == last || *lowerBoundIt > primeToRemove)
                {
                    relations[k].oddPrimeFactors.insert(lowerBoundIt, primeToRemove);
                }
                else
                {
                    relations[k].oddPrimeFactors.erase(lowerBoundIt);
                }

                //cout << "\t afterwards: ", relations[k].print();
            }

            //assert that no other has 1 at front
            for(k=k+1; k<relations.size(); k++)
            {
                auto start = relations[k].oddPrimeFactors.begin();
                auto last = relations[k].oddPrimeFactors.end();
                if(find(start, last, currentPrime) != last)
                    throw logic_error("asd");
            }

        }

        relations[i].oddPrimeFactors.erase(nextPrimeIterator+1, relations[i].oddPrimeFactors.end());

    }




    //cout << "last prime: " << currentPrime << endl;; 
    //comparator.minPrime = currentPrime;
    //sort(relations.begin()+i, relations.end(), comparator);
}

bool isNullPair(const pair<BigInt,BigInt>& pair) {return pair.first == 0 && pair.second == 0;}

pair<BigInt,BigInt> checkRelationCombination(const vector<Relation>& relations, BigInt& n)
{
    //find max Prime    
    uint32_t maxPrime = 0;
    for(const Relation& relation : relations)
    {
        uint32_t lastPrime = relation.primeFactorization.back();
        if(lastPrime > maxPrime)
            maxPrime = lastPrime;
    }

    BigInt a = 1;
    BigInt b = 1;
    vector<uint32_t> primeFactorTimes(maxPrime+1);
    for(const Relation& relation : relations)
    {
        a = (a * relation.a) % n;
        for(uint32_t prime : relation.primeFactorization)
            primeFactorTimes[prime]++;
    }

    for(uint32_t prime=0; prime <= maxPrime; prime++)
    {
        for(uint32_t i=0; i<primeFactorTimes[prime]/2; i++)
        {
            b = (b*prime) % n;
        }
    }

    BigInt sum = a+b;
    BigInt diff = (a>b)?(a-b):(b-a);
    BigInt p,q;
    mpz_gcd(p.get_mpz_t(), sum.get_mpz_t(), n.get_mpz_t());
    mpz_gcd(q.get_mpz_t(), diff.get_mpz_t(), n.get_mpz_t());

    if(p>1 && p<n && q>1 && q<n)
    {
        return pair<BigInt,BigInt>(p,q);
    }  

    return pair<BigInt,BigInt>(0,0);
}

pair<BigInt, BigInt> extractPrimeSolutions(vector<Relation>& relations, BigInt& n)
{
    //find max Prime    
    uint32_t maxPrime = 0;
    for(const Relation& relation : relations)
    {
        uint32_t lastPrime = relation.primeFactorization.back();
        if(lastPrime > maxPrime)
            maxPrime = lastPrime;
    }

    vector<bool> primeMask(maxPrime+1);
    vector<Relation> freeRelations;

    uniform_int_distribution<int> distribution(0, 1);
    mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);

    for(int i=0; i<1000; i++) //random 1000
    {
        vector<Relation> selectedRelations;
        for(auto relIter = relations.rbegin(); relIter != relations.rend(); relIter++)
        {
            const Relation& relation = *relIter;

            if((relation.dependsOnPrime == 0 && generator() == 1) || (relation.dependsOnPrime > 0 && primeMask[relation.dependsOnPrime]))
            {   
                selectedRelations.push_back(relation);
                for(uint32_t p : relation.oddPrimeFactors)
                    primeMask[p] = !primeMask[p];
            }
        }
        
        if(selectedRelations.empty())
            continue;

        pair<BigInt,BigInt> result = checkRelationCombination(selectedRelations, n);
        if(!isNullPair(result))
            return result;
    }

    return pair<BigInt,BigInt>(0,0);
}



void printRelation(pair<uint64_t, vector<uint32_t>>& rel, BigInt& sqrtN, vector<uint32_t> primes)
{
    cout << sqrtN + rel.first << "^2 = 1";
    for(size_t i=0; i<numberOfPrimes; i++)
    {
        if(rel.second[i] > 0)
        {
            cout << " * " << primes[i];
        }
    }
    cout << endl;
}


pair<BigInt, BigInt> factorize_qs(BigInt& n)
{

    vector<uint32_t> primes = createFirstNPrimes(numberOfPrimes);

    BigInt sqrtN = sqrt(n);
    BigInt remainder;

    BigInt x = sqrtN +1;

    vector<Relation> relations;


    auto start = high_resolution_clock::now();
    uint64_t lastRelationIndex = 1;


    vector<uint32_t> factors(numberOfPrimes);     
    for(uint64_t i=1; i<UINT64_C(10000000); i++, x++)
    {
        remainder = (x*x) % n;
        //mpz_pow_ui(remainder.get_mpz_t(), x.get_mpz_t(), 2);
        //remainder -= n;
   
        fill(factors.begin(), factors.end(), 0);
        if(getPrimeFactors(remainder, primes, factors))
        {
            cout << "relation found! (" << relations.size() << ")" << endl;
            Relation relation;
            relation.a = x;
            relation.dependsOnPrime = 0;

            BigInt a_test = 1;
            for(uint32_t k=0; k<numberOfPrimes; k++)
            {
                for(uint32_t j=0; j<factors[k]; j++)
                {
                    relation.primeFactorization.push_back(primes[k]);
                    a_test *= primes[k];
                }
                if(factors[k] % 2 == 1)
                {
                    relation.oddPrimeFactors.push_back(primes[k]);
                }
            }
            relation.print();
            cout << "a^2 (n) = " << ((relation.a*relation.a)%n) << " = " << a_test << " = product" << endl;

            if(relation.oddPrimeFactors.empty()) // could be square by itself
            {
                pair<BigInt,BigInt> result = checkRelationCombination(vector<Relation>({relation}), n);
                //if(!isNullPair(result))
                //    return result;
            }
            else
            {              
                relations.push_back(relation);
            }
        }

        auto end = high_resolution_clock::now();
        milliseconds elapsed = duration_cast<milliseconds>(end - start);
        if(elapsed.count() > 10000)
        {
            double relationCount = i - lastRelationIndex;
            lastRelationIndex = i;  
            std::cout << "relations per second: " << relationCount * 1000.0 / elapsed.count() << endl;
            start = end;
        }

        if(relations.size() > numberOfPrimes + 20)
        {
            cout << "Enough relations found!" << endl;
            break;
        }
    }


    //
    cout << "before gauss" << endl;
    gaussEliminiation(relations);
    cout << "after gauss" << endl;

    //
    cout << "before extractPrimeSolutions" << endl;
    auto pq = extractPrimeSolutions(relations, n);
    cout << "after extractPrimeSolutions" << endl;
    auto p = pq.first;
    auto q = pq.second;

    return pair<BigInt,BigInt>(p,q);
} 


TEST(BigIntTest, testFactorizationQuadraticSieve)
{
    BigInt p("1313839");
    BigInt q("1327901");
    BigInt n = p*q; 





    auto start = high_resolution_clock::now();

    auto pq = factorize_qs(n);


    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << "total time: " << elapsed.count() / 1000.0 << " seconds" << endl;  


    if(isNullPair(pq))
    {
        cout << "No Solution found :-(" << endl;
    }
    else
    {
        cout << "Solution found! :-D" << endl;
        cout << " N  = " << n << endl;
        cout << " p  = " << pq.first << endl;
        cout << " q  = " << pq.second << endl;
        cout << "p*q = " << pq.first * pq.second << endl;
    }

/*
    vector<Relation> relations;
    Relation r;

    r.a = 1;
    r.oddPrimeFactors = vector<uint32_t>({3, 5});
    r.dependsOnPrime = 0;
    relations.push_back(r);

    r.a = 2;
    r.oddPrimeFactors = vector<uint32_t>({2, 5});
    r.dependsOnPrime = 0;
    relations.push_back(r);

    r.a = 3;
    r.oddPrimeFactors = vector<uint32_t>({2, 3, 5});
    r.dependsOnPrime = 0;
    relations.push_back(r);

    r.a = 4;
    r.oddPrimeFactors = vector<uint32_t>({2, 3});
    r.dependsOnPrime = 0;
    relations.push_back(r);

    r.a = 5;
    r.oddPrimeFactors = vector<uint32_t>({5, 7});
    r.dependsOnPrime = 0;
    relations.push_back(r);

    gaussEliminiation(relations);

    for(size_t i=0; i<relations.size(); i++)
    {
        relations[i].print();
    }
    */
}


