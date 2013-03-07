#ifdef HAVE_CUDA

#include <elves/common-factorization/BigInt.h>
#include <elves/quadratic_sieve/cuda/Factorize.h>
#include <elves/quadratic_sieve/cuda/NumberHelper.h>
#include <elves/cuda-utils/Memory.h>
#include <gtest/gtest.h>
#include <functional>
#include <cstdlib>

extern void testAdd(PNumData left, PNumData right, PNumData result);
extern void testSub(PNumData left, PNumData right, PNumData result);
extern void testMul(PNumData left, PNumData right, PNumData result);
extern void testDiv(PNumData left, PNumData right, PNumData result);
extern void testSmallerThan(PNumData left, PNumData right, bool* result);
extern void testShiftLeft(PNumData left, uint32_t offset, PNumData result);
extern void testShiftRight(PNumData left, uint32_t offset, PNumData result);

using namespace std;

BigInt invokeShiftKernel(const BigInt& left, const uint32_t right, function<void (PNumData, uint32_t, PNumData)> kernelCall)
{
    CudaUtils::Memory<uint32_t> left_d(NumberHelper::BigIntToNumber(left));
    CudaUtils::Memory<uint32_t> out_d(NUM_FIELDS);

    kernelCall(left_d.get(), right, out_d.get());
    return NumberHelper::NumberToBigInt(out_d);
}

bool invokeBoolKernel(const BigInt& left, const BigInt& right, function<void (PNumData, PNumData, bool*)> kernelCall)
{
    bool outputData;

    CudaUtils::Memory<uint32_t> left_d(NumberHelper::BigIntToNumber(left));
    CudaUtils::Memory<uint32_t> right_d(NumberHelper::BigIntToNumber(right));
    CudaUtils::Memory<bool> out_d(1);

    kernelCall(left_d.get(), right_d.get(), out_d.get());
    out_d.transferTo(&outputData);

    return outputData;
}

BigInt invokeKernel(const BigInt& left, const BigInt& right, function<void (PNumData, PNumData, PNumData)> kernelCall)
{
    CudaUtils::Memory<uint32_t> left_d(NumberHelper::BigIntToNumber(left));
    CudaUtils::Memory<uint32_t> right_d(NumberHelper::BigIntToNumber(right));
    CudaUtils::Memory<uint32_t> out_d(NUM_FIELDS);

    kernelCall(left_d.get(), right_d.get(), out_d.get());

    return NumberHelper::NumberToBigInt(out_d);
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

    auto actual = invokeKernel(left, right, testAdd);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testAdditionSmallValues)
{
    BigInt left("1");
    BigInt right("1");
    BigInt expected("2");

    auto actual = invokeKernel(left, right, testAdd);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSubtraction)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("90108101410172790");

    auto actual = invokeKernel(left, right, testSub);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSubtractionSmallValues)
{
    BigInt left("2");
    BigInt right("1");
    BigInt expected("1");

    auto actual = invokeKernel(left, right, testSub);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSubtractionSmallValues2)
{
    BigInt left("1");
    BigInt right("1");
    BigInt expected("0");

    auto actual = invokeKernel(left, right, testSub);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testMultiplication)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("70873452463358713606126842179959");

    auto actual = invokeKernel(left, right,  testMul);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testMultiplicationSmallNumbers)
{
    BigInt left("5");
    BigInt right("8");
    BigInt expected("40");

    auto actual = invokeKernel(left, right,  testMul);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionSmallValues)
{
    BigInt left("6");
    BigInt right("2");
    BigInt expected("3");

    //auto actual = invokeKernel(left, right,  testDiv);

    //EXPECT_EQ(expected, actual);
}


TEST(CudaBigIntTest, testDivision)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("116");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionEqualValues)
{
    BigInt left("90887891231490623");
    BigInt right("90887891231490623");
    BigInt expected("1");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionEqualOperands)
{
    BigInt left("90887891231490623");
    BigInt expected("1");

    auto actual = invokeKernel(left, left,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThan)
{
    BigInt left("90887891231490623");
    BigInt right("7797822229821317833");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThanSmallValues)
{
    BigInt left("2");
    BigInt right("3");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThanSmallEqualValues)
{
    BigInt left("3");
    BigInt right("3");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testNotSmallerThan)
{
    BigInt left("90887891231490624");
    BigInt right("90887891231490623");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThanWithEqualOperands)
{
    BigInt left("90887891231490623");
    BigInt right("90887891231490623");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftLeft)
{
    BigInt left("1");
    uint32_t offset(32);
    BigInt expected("4294967296");

    auto actual = invokeShiftKernel(left, offset, testShiftLeft);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftLeftSmallValues)
{
    BigInt left("1");
    uint32_t offset(1);
    BigInt expected("2");

    auto actual = invokeShiftKernel(left, offset, testShiftLeft);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftLeftSmallValues2)
{
    BigInt left("2");
    uint32_t offset(1);
    BigInt expected("4");

    auto actual = invokeShiftKernel(left, offset, testShiftLeft);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftLeftBiggerNumber)
{
    BigInt left("1282943598234");
    uint32_t offset(23);
    BigInt expected("10762110931694518272");

    auto actual = invokeShiftKernel(left, offset, testShiftLeft);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftLeftWithBigShiftOffset)
{
    BigInt left("1282943598234");
    uint32_t offset(3333);
    BigInt expected("0");

    auto actual = invokeShiftKernel(left, offset, testShiftLeft);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRight)
{
    BigInt left("4294967296");
    uint32_t offset(32);
    BigInt expected("1");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRightSmallValues)
{
    BigInt left("2");
    uint32_t offset(1);
    BigInt expected("1");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRightSmallValues2)
{
    BigInt left("1");
    uint32_t offset(1);
    BigInt expected("0");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRightBiggerNumber)
{
    BigInt left("1301820391234234234342");
    uint32_t offset(33);
    BigInt expected("151551839806");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRightBiggerNumber2)
{
    BigInt left("2135987035920910082395021706169552114602704522356652769947041607822219725780640550022962086936575");
    uint32_t offset(33);
    BigInt expected("248661618204893321077691124073410420050228075398673858720231988446579748506266687766527");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testShiftRightWithBigShiftOffset)
{
    BigInt left("1");
    uint32_t offset(33333);
    BigInt expected("0");

    auto actual = invokeShiftKernel(left, offset, testShiftRight);

    EXPECT_EQ(expected, actual);
}


#endif /* HAVE_CUDA */
