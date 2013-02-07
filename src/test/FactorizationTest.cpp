#include "FactorizationTest.h"
#include "Utilities.h"
#include "common/SchedulerFactory.h"
#include "elves/common-factorization/BigInt.h"
#include "elves/quadratic_sieve/smp/QuadraticSieve.h"
#include "elves/quadratic_sieve/cuda/CudaQuadraticSieveElf.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

typedef vector<uint64_t> PrimeList;

const PrimeList SMALL_PRIMES = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };
const PrimeList PRIMES_BELOW_100 = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };

INSTANTIATE_TEST_CASE_P(
    PrimePairs,
    FactorizationTest,
    testing::Values(
        //make_pair(BigInt("37975227936943673922808872755445627854565536638199"),
        //          BigInt("40094690950920881030683735292761468389214899724061"))
        // The large pair takes too long at the moment
        //make_pair(BigInt("551226983117"), BigInt("554724632351"))
        make_pair(BigInt("15485863"), BigInt("15534733")),
        make_pair(BigInt("1313839"), BigInt("1327901")),
        make_pair(BigInt("547"), BigInt("719"))
    )
);

void FactorizationTest::SetUp()
{
    auto inputPair = GetParam();
    p = inputPair.first;
    q = inputPair.second;
    product = p*q;
}

/*
TEST_P(FactorizationTest, testFactorizationCuda)
{
    unique_ptr<CudaFactorizationElf> elf(new CudaFactorizationElf());
    auto actual = elf->factorize(product);
    EXPECT_EQ(p, actual.first);
    EXPECT_EQ(q, actual.second);
}*/

TEST_P(FactorizationTest, testFactorizationFermat)
{
    BigInt n = sqrt(product);

    BigInt x = n;
    BigInt xx;
    BigInt y;
    BigInt yy;

    auto start = high_resolution_clock::now();

    for(uint64_t i=0; ;i++)
    {
        x = x + 1;

        mpz_powm_ui(xx.get_mpz_t(), x.get_mpz_t(), 2, product.get_mpz_t());

        y = sqrt(xx);

        mpz_pow_ui(yy.get_mpz_t(), y.get_mpz_t(), 2);

        if (xx == yy)
        {
            EXPECT_EQ(p, x-y);
            EXPECT_EQ(q, x+y);
            break;
        }
    }

    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << elapsed.count() / 1000.0 << '\n';

}

TEST_P(FactorizationTest, testFactorizationPollardRho)
{
    function<BigInt(BigInt)> f ([=](BigInt x){
        BigInt result = (x*x+123) % product;
        return result;
    });

    BigInt x(2);
    BigInt y(x);
    BigInt d(1);
    BigInt absdiff;

    auto start = high_resolution_clock::now();

    while(d == 1)
    {
        x = f(x);
        y = f(y);
        y = f(y);
        absdiff = abs(x-y);
        mpz_gcd(d.get_mpz_t(), absdiff.get_mpz_t(), product.get_mpz_t());
        if(1 < d && d < product)
        {
            BigInt pp = d;
            BigInt qq = product / d;

            if(pp > qq)
            {
                BigInt tmp(pp);
                pp = qq;
                qq = tmp;
            }

            EXPECT_EQ(p, pp);
            EXPECT_EQ(q, qq);

            break;
        }
    }

    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << elapsed.count() / 1000.0 << '\n';

}
/*
TEST_P(FactorizationTest, testFactorizationQuadraticSieve)
{
    QuadraticSieve qs(product);

    auto start = high_resolution_clock::now();

    auto pq = qs.factorize();

    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << "total time: " << elapsed.count() / 1000.0 << " seconds" << endl;


    BigInt actualP = pq.first;
    BigInt actualQ = pq.second;

    if(actualP > actualQ)
        swap(actualP, actualQ);

    ASSERT_EQ(p, actualP);
    ASSERT_EQ(q, actualQ);
}

TEST(QuadraticSieveTest, testModularSquareRoot)
{
    BigInt primeMod("104729");

    vector<string> roots = {"2", "12321", "4563", "34513", "13", "567856", "103729"};

    for(const string& rootstring : roots)
    {
        BigInt expectedRoot(rootstring);
        BigInt n = (expectedRoot*expectedRoot) % primeMod;
        BigInt root = QuadraticSieve::rootModPrime(n, primeMod);
        ASSERT_NE(0, root);
        ASSERT_EQ(n, (root*root)%primeMod);
    }
}

TEST(QuadraticSieveTest, testModularSquareRoot2)
{
    BigInt primeMod("2909");

    vector<string> roots = {"305779185551528709018067"};

    for(const string& rootstring : roots)
    {
        BigInt expectedRoot(rootstring);
        BigInt n = (expectedRoot*expectedRoot) % primeMod;
        BigInt root = QuadraticSieve::rootModPrime(n, primeMod);
        ASSERT_NE(0, root);
        ASSERT_EQ(n, (root*root)%primeMod);
    }
}


TEST(QuadraticSieveTest, testModularSquareRootInvalid)
{
    BigInt primeMod("7");
    BigInt n("3");

    ASSERT_THROW(QuadraticSieve::rootModPrime(n, primeMod), logic_error);
}

TEST(QuadraticSieveTest, testExtensiveSquareRooting)
{
    vector<string> primeStrings = {"2", "3", "5", "7", "11", "13", "71", "229", "541"};
    BigInt threshold("10000");

    BigInt primePower;
    for(const string& primeString : primeStrings)
    {
        BigInt prime(primeString);
        BigInt primePower = prime;
        for(uint32_t power=1; primePower < threshold; power++, primePower*=prime)
        {
            map<BigInt, vector<BigInt>> rootsPerResidue;
            for(BigInt x=0; x<primePower; ++x)
            {
                BigInt residue = (x*x) % primePower;
                rootsPerResidue[residue].push_back(x);
            }

            for (auto& rpr: rootsPerResidue) {
                const BigInt& residue = rpr.first;
                const vector<BigInt>& expectedRoots = rpr.second;

                auto actualRoots = QuadraticSieve::squareRootsModPrimePower(
                                    residue, prime, power);
                sort(actualRoots.begin(), actualRoots.end());

                ASSERT_EQ(expectedRoots.size(), actualRoots.size()) << "Vectors x and y are of unequal length";
                for (size_t i = 0; i < actualRoots.size(); ++i) {
                  ASSERT_EQ(expectedRoots[i], actualRoots[i]) << "Vectors x and y differ at index " << i;
                }
            }
        }
    }
}
*/
