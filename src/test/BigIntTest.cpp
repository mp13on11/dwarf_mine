#include "factorize/BigInt.h"

#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>

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
    EXPECT_THROW(b - a, logic_error);

    a -= b;

    EXPECT_EQ(BigInt(2222), a);
    EXPECT_EQ(BigInt(7777), b);
}

TEST(BigIntTest, regressionTestSubtractionWithZeroResult)
{
    BigInt a(1234);

    EXPECT_EQ(BigInt::ZERO, a - a);
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
    BigInt a(BigInt::MAX_ITEM);
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

    EXPECT_EQ(value, a.toString());
}

TEST(BigIntTest, testLargeMultiplication)
{
    BigInt a("987654321098765432109876543210987654321");
    a *= 100000000;

    EXPECT_EQ("98765432109876543210987654321098765432100000000", a.toString());
}

TEST(BigIntTest, testMultiplicationWithZero)
{
    BigInt a("89739847514365781347561873658761230102357816");

    EXPECT_EQ(BigInt::ZERO, a * BigInt::ZERO);
    EXPECT_EQ(BigInt::ZERO, BigInt::ZERO * a);
}

TEST(BigIntTest, testMultiplcaitionWithOne)
{
    BigInt a("12340898919834501231239487180439512934801234");

    EXPECT_EQ(a, a * BigInt::ONE);
    EXPECT_EQ(a, BigInt::ONE * a);
}

TEST(BigIntTest, testLargeModulo)
{
    BigInt a("234981234987123478913489712834");
    a %= 100000000;

    EXPECT_EQ("89712834", a.toString());
}

TEST(BigIntTest, testLargeDivision)
{
    BigInt a("81234897128357891239878923649612356981237598123");
    a /= 1000000000;

    EXPECT_EQ("81234897128357891239878923649612356981", a.toString());
}

TEST(BigIntTest, testLeftShift)
{
    BigInt a(1);

    a <<= 4;
    EXPECT_EQ("16", a.toString());

    a <<= 7;
    EXPECT_EQ("2048", a.toString());

    a <<= 32;
    EXPECT_EQ("8796093022208", a.toString());

    a <<= 53;
    EXPECT_EQ("79228162514264337593543950336", a.toString());
}

TEST(BigIntTest, testRightShift)
{
    BigInt a("79228162514264337593543950336");

    a >>= 53;
    EXPECT_EQ("8796093022208", a.toString());

    a >>= 32;
    EXPECT_EQ("2048", a.toString());

    a >>= 7;
    EXPECT_EQ("16", a.toString());

    a >>= 4;
    EXPECT_EQ("1", a.toString());
}

TEST(BigIntTest, testLargeAddition)
{
    BigInt a("10101010101010101010101010101010101010101010101010101010101");
    BigInt b("02020202020202020202020202020202020202020202020202020202020");
    BigInt c = a + b;

    EXPECT_EQ(
            "12121212121212121212121212121212121212121212121212121212121",
            c.toString()
        );
}

TEST(BigIntTest, testLargeSubtraction)
{
    BigInt a("12121212121212121212121212121212121212121212121212121212121");
    BigInt b("02020202020202020202020202020202020202020202020202020202020");
    BigInt c = a - b;

    EXPECT_EQ(
            "10101010101010101010101010101010101010101010101010101010101",
            c.toString()
        );
}

TEST(BigIntTest, testZeroDivisionOperator)
{
    BigInt a("24512354124");
    BigInt b = BigInt::ZERO / a;

    EXPECT_EQ(BigInt::ZERO, b);
}

TEST(BigIntTest, testArithmeticAssignmentAdditionOperator)
{
    BigInt a("1234567892342345");
    
    BigInt expected = a + a;
    a += a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentSubtractionOperator)
{
    BigInt a("123456789000000");
    
    BigInt expected = a - a;
    a -= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentMultiplyOperator)
{
    BigInt a("123456789000000");
    
    BigInt expected = a * a;
    a *= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentDivisionOperator)
{
    BigInt a("123456789000000");
    
    BigInt expected = a / a;
    a /= a;

    EXPECT_EQ(expected, a);
}

TEST(BigIntTest, testArithmeticAssignmentRemainderOperator)
{
    BigInt a("123456789000000");
    
    BigInt expected = a % a;
    a %= a;

    EXPECT_EQ(expected, a);
}