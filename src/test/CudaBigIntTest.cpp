/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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
extern void testMod(PNumData left, PNumData right, PNumData result);
extern void testSmallerThan(PNumData left, PNumData right, bool* result);
extern void testLargerThan(PNumData left, PNumData right, bool* result);
extern void testLargerEqual(PNumData left, PNumData right, bool* result);
extern void testEqual(PNumData left, PNumData right, bool* result);
extern void testShiftLeft(PNumData left, uint32_t offset, PNumData result);
extern void testShiftRight(PNumData left, uint32_t offset, PNumData result);
extern void testModPow(PNumData base, PNumData exponent, PNumData mod, PNumData result);
extern void testCudaPow(int b, int e, int* result);

using namespace std;

int invokeCudaPowKernel(int b, int e, function<void (int, int, int*)> kernelCall)
{
    CudaUtils::Memory<int> out_d(1);

    kernelCall(b, e, out_d.get());
    int out;
    out_d.transferTo(&out);
    return out;
}

BigInt invokeModPowKernel(const BigInt& base, const BigInt& exponent, const BigInt& mod, function<void (PNumData, PNumData, PNumData, PNumData)> kernelCall)
{
    CudaUtils::Memory<uint32_t> base_d(NumberHelper::BigIntToNumber(base));
    CudaUtils::Memory<uint32_t> exponent_d(NumberHelper::BigIntToNumber(exponent));
    CudaUtils::Memory<uint32_t> mod_d(NumberHelper::BigIntToNumber(mod));

    CudaUtils::Memory<uint32_t> out_d(NUM_FIELDS);

    kernelCall(base_d.get(), exponent_d.get(), mod_d.get(), out_d.get());
    return NumberHelper::NumberToBigInt(out_d);
}

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

TEST(CudaBigIntTest, testDivision)
{
    BigInt left("90887891231490623");
    BigInt right("779789821317833");
    BigInt expected("116");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}


TEST(CudaBigIntTest, testDivision2)
{
    BigInt left("1");
    BigInt right("2");
    BigInt expected("0");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionSmallValues)
{
    BigInt left("12");
    BigInt right("2");
    BigInt expected("6");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionSmallValues2)
{
    BigInt left("4");
    BigInt right("2");
    BigInt expected("2");

    auto actual = invokeKernel(left, right,  testDiv);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testDivisionSmallValues3)
{
    BigInt left("7");
    BigInt right("4");
    BigInt expected("1");

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

TEST(CudaBigIntTest, testModulo)
{
    BigInt left("100");
    BigInt right("6");
    BigInt expected("4");

    auto actual = invokeKernel(left, right,  testMod);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testModulo2)
{
    BigInt left("100");
    BigInt right("2");
    BigInt expected("0");

    auto actual = invokeKernel(left, right,  testMod);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testModulo3)
{
    BigInt left("8");
    BigInt right("100");
    BigInt expected("8");

    auto actual = invokeKernel(left, right,  testMod);

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

TEST(CudaBigIntTest, testSmallerThan2)
{
    BigInt left ("90887891231490623");
    BigInt right("779789821317833");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThan3)
{
    BigInt left ("628");
    BigInt right("886");
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

TEST(CudaBigIntTest, testSmallerThanSmallValues2)
{
    BigInt left("4");
    BigInt right("2");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testSmallerThanSmallValueAndLargeValue)
{
    BigInt left("2");
    BigInt right("2");
    right <<= 32;

    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testNotSmallerThanSmallValues)
{
    BigInt left("3");
    BigInt right("2");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testNotSmallerThanWithZeroAsOperand)
{
    BigInt left("0");
    BigInt right("0");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testSmallerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testNotSmallerThanSmallEqualValues)
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

TEST(CudaBigIntTest, testLargerThan)
{
    BigInt left ("798798797897897897987");
    BigInt right("4564654654654656");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerThan2)
{
    BigInt left ("628");
    BigInt right("886");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testLargerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerThanSmallValues)
{
    BigInt left("2");
    BigInt right("1");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testNotLargerThanSmallValues)
{
    BigInt left("1");
    BigInt right("2");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testLargerThan);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerWithEqualOperands)
{
    BigInt left("2");
    BigInt right("2");
    bool expected(false);

    auto actual = invokeBoolKernel(left, right, testLargerThan);

    EXPECT_EQ(expected, actual);
}


TEST(CudaBigIntTest, testLargerEqual)
{
    BigInt left("8589934592");
    BigInt right("8589934592");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerEqual);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerEqualWithSmallEqualOperands)
{
    BigInt left("2");
    BigInt right("2");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerEqual);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerEqualActuallyLarger)
{
    BigInt left("8589934593");
    BigInt right("8589934592");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerEqual);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testLargerEqualSmallAndLargeValue)
{
    BigInt left("2");
    left <<= 32;
    BigInt right("2");

    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testLargerEqual);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testEqual)
{
    BigInt left("222222222222222222");
    BigInt right("222222222222222222");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testEqual);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testEqualSmallValues)
{
    BigInt left("0");
    BigInt right("0");
    bool expected(true);

    auto actual = invokeBoolKernel(left, right, testEqual);

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

TEST(CudaBigIntTest, testModPow)
{
    BigInt base("2");
    BigInt exponent("3");
    BigInt mod("100");
    BigInt expected("8");

    auto actual = invokeModPowKernel(base, exponent, mod, testModPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testModPow2)
{
    BigInt base("55");
    BigInt exponent("80");
    BigInt mod("13");
    BigInt expected("9");

    auto actual = invokeModPowKernel(base, exponent, mod, testModPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testModPow3)
{
    BigInt base("55");
    BigInt exponent("81");
    BigInt mod("17");
    BigInt expected("4");

    auto actual = invokeModPowKernel(base, exponent, mod, testModPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testModPow4)
{
    BigInt base("123123");
    BigInt exponent("2");
    BigInt mod("5");
    BigInt expected("4");

    auto actual = invokeModPowKernel(base, exponent, mod, testModPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testCudaPow)
{
    int base = 9;
    int exp = 2;

    int expected(81);

    auto actual = invokeCudaPowKernel(base, exp, testCudaPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testCudaPow2)
{
    int base = 123;
    int exp = 2;

    int expected(15129);

    auto actual = invokeCudaPowKernel(base, exp, testCudaPow);

    EXPECT_EQ(expected, actual);
}

TEST(CudaBigIntTest, testCudaPow3)
{
    int base = 81;
    int exp = 4;

    int expected(43046721);

    auto actual = invokeCudaPowKernel(base, exp, testCudaPow);

    EXPECT_EQ(expected, actual);
}

#endif /* HAVE_CUDA */

