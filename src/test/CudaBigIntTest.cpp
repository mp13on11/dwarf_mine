#include <elves/factorize/BigInt.h>
#include <elves/factorize/cuda/Factorize.h>
#include <elves/cuda-utils/Memory.h>
#include <gtest/gtest.h>
#include <functional>
#include <cstdlib>

extern void testAdd(PNumData left, PNumData right, PNumData result);
extern void testSub(PNumData left, PNumData right, PNumData result);
extern void testMul(PNumData left, PNumData right, PNumData result);
extern void testDiv(PNumData left, PNumData right, PNumData result);

using namespace std;

BigInt invokeKernel(const BigInt& left, const BigInt& right, function<void (PNumData, PNumData, PNumData)> kernelCall)
{
    NumData leftData;
    NumData rightData;
    NumData outputData;
    memset(leftData, 0, sizeof(uint32_t) * NUM_FIELDS);
    memset(rightData, 0, sizeof(uint32_t) * NUM_FIELDS);
    mpz_export(leftData, nullptr, -1, sizeof(uint32_t), 0, 0, left.get_mpz_t());
    mpz_export(rightData, nullptr, -1, sizeof(uint32_t), 0, 0, right.get_mpz_t());

    CudaUtils::Memory<uint32_t> left_d(NUM_FIELDS);
    CudaUtils::Memory<uint32_t> right_d(NUM_FIELDS);
    CudaUtils::Memory<uint32_t> out_d(NUM_FIELDS);

    left_d.transferFrom(leftData);
    right_d.transferFrom(rightData);
    kernelCall(left_d.get(), right_d.get(), out_d.get());
    out_d.transferTo(outputData);

    BigInt mpzResult;
    mpz_import(mpzResult.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, outputData);
    return mpzResult;
}

TEST(CudaBigIntTest, testMpzConversion)
{
    BigInt start("1231490623");
    NumData test;
    memset(test, 0, sizeof(uint32_t) * NUM_FIELDS);
    mpz_export(test, nullptr, -1, sizeof(uint32_t), 0, 0, start.get_mpz_t());
    BigInt compare;
    mpz_import(compare.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, test);

    EXPECT_EQ(start, compare);
}

TEST(CudaBigIntTest, testAddition)
{
    BigInt left("12314906231232141243");
    BigInt right("21317833214213");
    BigInt expected("12314927549065355456");

    auto actual = invokeKernel(left, right, [](PNumData l, PNumData r, PNumData o) { testAdd(l, r, o); });

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSubtraction)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("90108101410172790");

    auto actual = invokeKernel(left, right, [](PNumData l, PNumData r, PNumData o) { testSub(l, r, o); });

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testMultiplication)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("70873452463358713606126842179959");

    auto actual = invokeKernel(left, right, [](PNumData l, PNumData r, PNumData o) { testMul(l, r, o); });

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivision)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("116");

    auto actual = invokeKernel(left, right, [](PNumData l, PNumData r, PNumData o) { testDiv(l, r, o); });

    EXPECT_EQ(expected, actual);
}
