#include "common-factorization/BigInt.h"

#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <functional>

using namespace std;
using namespace chrono;
using namespace testing;

TEST(BigIntTest, testSmallComparison)
{
    BigInt a(6174);

    EXPECT_EQ(BigInt(6174), a);
    EXPECT_NE(BigInt(1337), a);
}

TEST(BigIntTest, testSmallAddition)
{
    BigInt a(1234);
    BigInt b(7654);

    EXPECT_EQ(BigInt(8888), a + b);
    EXPECT_EQ(BigInt(8888), b + a);

    a += b;

    EXPECT_EQ(BigInt(8888), a);
    EXPECT_EQ(BigInt(7654), b);
}

TEST(BigIntTest, testSmallSubtraction)
{
    BigInt a(9999);
    BigInt b(7777);

    EXPECT_EQ(BigInt(2222), a - b);
    EXPECT_EQ(BigInt(-2222), b - a);

    a -= b;

    EXPECT_EQ(BigInt(2222), a);
    EXPECT_EQ(BigInt(7777), b);
}

TEST(BigIntTest, regressionTestSubtractionWithZeroResult)
{
    BigInt a(1234);

    EXPECT_EQ(0, a - a);
}

TEST(BigIntTest, testSmallMultiplication)
{
    BigInt a(1234);
    BigInt b(25);

    EXPECT_EQ(BigInt(30850), a * b);
    EXPECT_EQ(BigInt(30850), b * a);

    a *= b;

    EXPECT_EQ(BigInt(30850), a);
    EXPECT_EQ(BigInt(25), b);
}

TEST(BigIntTest, testAdditionSubtractionRoundTrip)
{
    BigInt a("2346871365871234");
    BigInt b(12345);
    a += a;

    EXPECT_EQ(a, (a + b) - b);
    EXPECT_EQ(a, (a - b) + b);
}

TEST(BigIntTest, testParsingRoundTrip)
{
    string value = "1234567890123456789012345678901234567890123456789012345678";
    istringstream read(value);

    BigInt a;
    read >> a;

    ostringstream write;
    write << a;

    EXPECT_EQ(value, write.str());
}

TEST(BigIntTest, testStringRoundTrip)
{
    string value = "987654321098765432109876543210987654321";
    BigInt a(value);

    EXPECT_EQ(value, a.get_str());
}

TEST(BigIntTest, testLargeMultiplication)
{
    BigInt a("987654321098765432109876543210987654321");
    a *= 100000000;

    EXPECT_EQ("98765432109876543210987654321098765432100000000", a.get_str());
}

TEST(BigIntTest, testMultiplicationWithZero)
{
    BigInt a("89739847514365781347561873658761230102357816");

    EXPECT_EQ(0, a * 0);
    EXPECT_EQ(0, 0 * a);
}

TEST(BigIntTest, testMultiplcaitionWithOne)
{
    BigInt a("12340898919834501231239487180439512934801234");

    EXPECT_EQ(a, a * 1);
    EXPECT_EQ(a, 1 * a);
}

TEST(BigIntTest, testLargeModulo)
{
    BigInt a("234981234987123478913489712834");
    a %= 100000000;

    EXPECT_EQ("89712834", a.get_str());
}

TEST(BigIntTest, testLargeDivision)
{
    BigInt a("81234897128357891239878923649612356981237598123");
    a /= 1000000000;

    EXPECT_EQ("81234897128357891239878923649612356981", a.get_str());
}

TEST(BigIntTest, testLeftShift)
{
    BigInt a(1);

    a <<= 4;
    EXPECT_EQ("16", a.get_str());

    a <<= 7;
    EXPECT_EQ("2048", a.get_str());

    a <<= 32;
    EXPECT_EQ("8796093022208", a.get_str());

    a <<= 53;
    EXPECT_EQ("79228162514264337593543950336", a.get_str());
}

TEST(BigIntTest, testRightShift)
{
    BigInt a("79228162514264337593543950336");

    a >>= 53;
    EXPECT_EQ("8796093022208", a.get_str());

    a >>= 32;
    EXPECT_EQ("2048", a.get_str());

    a >>= 7;
    EXPECT_EQ("16", a.get_str());

    a >>= 4;
    EXPECT_EQ("1", a.get_str());
}

TEST(BigIntTest, testLargeAddition)
{
    BigInt a("10101010101010101010101010101010101010101010101010101010101");
    BigInt b("2020202020202020202020202020202020202020202020202020202020");
    BigInt c = a + b;

    EXPECT_EQ(
            "12121212121212121212121212121212121212121212121212121212121",
            c.get_str()
        );
}

TEST(BigIntTest, testLargeSubtraction)
{
    BigInt a("12121212121212121212121212121212121212121212121212121212121");
    BigInt b("2020202020202020202020202020202020202020202020202020202020");
    BigInt c = a - b;

    EXPECT_EQ(
            "10101010101010101010101010101010101010101010101010101010101",
            c.get_str()
        );
}

TEST(BigIntTest, testZeroDivisionOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");
    BigInt b = 0 / a;

    EXPECT_EQ(0, b);
}

TEST(BigIntTest, testArithmeticAssignmentAdditionOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");

    BigInt expected = a + a;
    a += a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentSubtractionOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");

    BigInt expected = a - a;
    a -= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentMultiplyOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");

    BigInt expected = a * a;
    a *= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentDivisionOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");

    BigInt expected = a / a;
    a /= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentRemainderOperator)
{
    BigInt a("123456789346545873246563762452465746764521398967478050785059870");

    BigInt expected = a % a;
    a %= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testMediumLogarithm)
{
    BigInt x("5512269831137356765576856757");
    uint32_t expected = 386524848;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}

TEST(BigIntTest, testLargeLogarithm)
{
    // x = 2^1024 - 3^566 + 754345635456
    BigInt x("179769313486231590772930519078902473360674050183465318615124150632640673711093104016960323758140143493142850675912072669434503023118988778667670255207828346892449845296516189112513154426849212248606207767356019216813220245023618250939644892482318865089392246229406300098045608631491456495938256080472342279143");
    uint32_t expected = 4294967295u;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}

TEST(BigIntTest, testSmallLogarithm)
{
    BigInt x("5687376457");
    uint32_t expected = 135916908;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}

TEST(BigIntTest, testTinyLogarithm)
{
    BigInt x("11");
    uint32_t expected = 14509907;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}

TEST(BigIntTest, test2Logarithm)
{
    BigInt x("2");
    uint32_t expected = 4194304;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}

TEST(BigIntTest, testOneLogarithm)
{
    BigInt x("1");
    uint32_t expected = 0;
    uint32_t value = log_2_22(x);
    ASSERT_EQ(expected, value);
}



