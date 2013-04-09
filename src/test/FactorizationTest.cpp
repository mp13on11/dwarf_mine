#include "FactorizationTest.h"
#include "Utilities.h"
#include "common/SchedulerFactory.h"
#include "elves/common-factorization/BigInt.h"
#include "elves/quadratic_sieve/QuadraticSieve.h"
#include "elves/quadratic_sieve/smp/SmpQuadraticSieveElf.h"
#include "elves/quadratic_sieve/cuda/CudaQuadraticSieveElf.h"

#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

// Does not terminate fast enough
//#define RUN_CUDA_QUADRATIC_SIEVE

typedef vector<uint64_t> PrimeList;

const PrimeList SMALL_PRIMES = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };
const PrimeList PRIMES_BELOW_100 = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };

INSTANTIATE_TEST_CASE_P(
    PrimePairs,
    FactorizationTest,
    testing::Values(
        //make_pair(BigInt("37975227936943673922808872755445627854565536638199"),
        //          BigInt("40094690950920881030683735292761468389214899724061"))
        //make_pair(BigInt("9499938415355683567"), BigInt("33172199367355317419")),
        make_pair(BigInt("551226983117"), BigInt("554724632351")),
        make_pair(BigInt("4231231247"), BigInt("4231231253")),
        make_pair(BigInt("15485863"), BigInt("15534733")),
        make_pair(BigInt("1313839"), BigInt("1327901")),
        make_pair(BigInt("3001"), BigInt("3011")),
        make_pair(BigInt("1009"), BigInt("1013")),
        make_pair(BigInt("547"), BigInt("719"))
    	//make_pair(BigInt("13"), BigInt("11")) // too small
    )
);

void FactorizationTest::SetUp()
{
    auto inputPair = GetParam();
    p = inputPair.first;
    q = inputPair.second;
    product = p*q;
}

TEST_P(FactorizationTest, testFactorizationQuadraticSieve)
{
    using namespace std::placeholders;
    auto start = high_resolution_clock::now();

    unique_ptr<QuadraticSieveElf> elf(new SmpQuadraticSieveElf());
    BigInt actualP, actualQ;
    tie(actualP, actualQ) = QuadraticSieveHelper::factor(product, bind(&QuadraticSieveElf::sieveSmoothSquares, elf.get(), _1, _2, _3, _4));


    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << "total time: " << elapsed.count() / 1000.0 << " seconds" << endl;

    if(actualP > actualQ)
        swap(actualP, actualQ);

    ASSERT_EQ(p, actualP);
    ASSERT_EQ(q, actualQ);
}

#ifdef RUN_CUDA_QUADRATIC_SIEVE
TEST_P(FactorizationTest, testFactorizationCudaQuadraticSieve)
{
    using namespace std::placeholders;
    auto start = high_resolution_clock::now();

    unique_ptr<QuadraticSieveElf> elf(new CudaQuadraticSieveElf());
    BigInt actualP, actualQ;
    tie(actualP, actualQ) = QuadraticSieveHelper::factor(product, bind(&QuadraticSieveElf::sieveSmoothSquares, elf.get(), _1, _2, _3, _4));


    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << "total time: " << elapsed.count() / 1000.0 << " seconds" << endl;

    if(actualP > actualQ)
        swap(actualP, actualQ);

    ASSERT_EQ(p, actualP);
    ASSERT_EQ(q, actualQ);
}
#endif

TEST(QuadraticSieveTest, testModularSquareRoot)
{
    BigInt primeMod("104729");

    vector<string> roots = {"2", "12321", "4563", "34513", "13", "567856", "103729"};

    for(const string& rootstring : roots)
    {
        BigInt expectedRoot(rootstring);
        BigInt n = (expectedRoot*expectedRoot) % primeMod;
        BigInt root = QuadraticSieveHelper::rootModPrime(n, primeMod);
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
        BigInt root = QuadraticSieveHelper::rootModPrime(n, primeMod);
        ASSERT_NE(0, root);
        ASSERT_EQ(n, (root*root)%primeMod);
    }
}


TEST(QuadraticSieveTest, testModularSquareRootInvalid)
{
    BigInt primeMod("7");
    BigInt n("3");

    ASSERT_THROW(QuadraticSieveHelper::rootModPrime(n, primeMod), logic_error);
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

                auto actualRoots = QuadraticSieveHelper::squareRootsModPrimePower(
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
