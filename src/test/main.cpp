#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <matrix/MatrixHelper.h>
#include <matrix/Matrix.h>
#include <gtest/gtest.h>

#include "MatrixMultiplyTest.h"

using namespace std;

::testing::AssertionResult AreMatricesEquals(Matrix<float> expected, Matrix<float>actual, float delta)
{
    int erroneous = 0;

    if(expected.rows() != actual.rows())
    {
        return ::testing::AssertionFailure() << "different number of rows: " << expected.rows() << " != " << actual.rows();
    }
    if(expected.columns() != actual.columns())
    {
        return ::testing::AssertionFailure() << "different number of columns: " << expected.columns() << " != " << actual.columns();
    }
    for(size_t y = 0; y<expected.rows(); y++)
    {
        for(size_t x = 0; x<expected.columns(); x++)
        {
            float expectedVal = expected(y,x);
            float actualVal = actual(y,x);
            float maxVal = max(fabs(expectedVal), fabs(actualVal));
            float error = fabs(expectedVal - actualVal);
            error = (maxVal < 1e-10) ? error : error / maxVal;
            if(error > delta)
            {
                ++erroneous;

                cout
                << "The relative error at (" << y << "," << x << ") is " << error << ", which exceeds " << delta << ", where" << endl
                << "expected(y,x) = " << expectedVal << " and" << endl
                << "actual(y,x) = " << actualVal << ".";
            }
        }
    }

    if (erroneous > 0)
    {
        // dump matrices
        MatrixHelper::writeMatrixTo("dump_expected.txt", expected);
        MatrixHelper::writeMatrixTo("dump_actual.txt", actual);

        // return failure
        return ::testing::AssertionFailure() 
            << erroneous << "/" << expected.rows()*expected.columns() << " are wrong";
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult AreMatricesEquals(Matrix<float> a, Matrix<float>b)
{
    return AreMatricesEquals(a, b, 1e-2);
}

TEST_P(MatrixMultiplyTest, SingleElementMatrixTest) {
    initRandom(0);
    auto left = createRandomMatrix(1, 1);
    auto right = createRandomMatrix(1, 1);;

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, SmallSquareMatricesTest) {
    initRandom(12345);
    auto left = createRandomMatrix(5, 5);
    auto right = createRandomMatrix(5, 5);

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, MediumShrinkingRectangularMatricesTest) {
    initRandom(333);
    auto left = createRandomMatrix(30, 100);
    auto right = createRandomMatrix(100, 40);

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}


TEST_P(MatrixMultiplyTest, MediumExpandingRectangularMatricesTest) {
    initRandom(4567);
    auto left = createRandomMatrix(110, 20);
    auto right = createRandomMatrix(20, 130);

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, PrimeRectangularMatricesTest) {
    initRandom(6543452);
    auto left = createRandomMatrix(67, 83);
    auto right = createRandomMatrix(83, 109);

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, BiggerPrimeRectangularMatricesTest) {
    initRandom(73653);
    auto left = createRandomMatrix(383, 269);
    auto right = createRandomMatrix(269, 193);

    auto expected = executeMultiplication(*referenceImplementation, left, right);
    auto actual = executeMultiplication(*currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}


INSTANTIATE_TEST_CASE_P(MultiplePlatforms,
                        MatrixMultiplyTest,
                        ::testing::Values("cuda", "smp"));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
