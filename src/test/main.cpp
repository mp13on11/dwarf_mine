#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cstdio>
#include <functional>
#include "tools/MatrixHelper.h"
#include "tools/Matrix.h"
#include "gtest/gtest.h"
#include <sstream>
#include <boost/algorithm/string/join.hpp>

using namespace std;


    

class MatrixMultiplyTest : public ::testing::TestWithParam<const char*>
{
private:



protected:

    virtual void SetUp() {
        //currentImplementation = GetParam();

        inputAFile = "a.txt";
        inputBFile = "b.txt";
        outputFile = "c.txt";

        referenceImplementation = "build/src/gold/gold";
        currentImplementation = "build/src/" + string(GetParam());
    }

    virtual void TearDown() {
        remove(inputAFile.c_str());
        remove(inputBFile.c_str());
        remove(outputFile.c_str());
    }

    Matrix<float> createRandomMatrix(size_t rows, size_t columns)
    {        
        Matrix<float> m(rows, columns);
        MatrixHelper::fill(m, generator);
        return m;
    }

    void initRandom(uint seed)
    {
        auto distribution = uniform_real_distribution<float> (-100, +100);
        auto engine = mt19937(seed);
        generator = bind(distribution, engine);
    }

    Matrix<float> executeMultiplication(string implementation, Matrix<float> a, Matrix<float> b)
    {
        MatrixHelper::writeMatrixTo(inputAFile, a);
        MatrixHelper::writeMatrixTo(inputBFile, b);
        startProcess({implementation, inputAFile, inputBFile, outputFile});
        return MatrixHelper::readMatrixFrom(outputFile);
    }

    void startProcess(initializer_list<string> args)
    {
        system(boost::algorithm::join(args, " ").c_str());
    }

    function<float()> generator;

    string inputAFile;
    string inputBFile;
    string outputFile;

    string referenceImplementation;
    string currentImplementation;
};

::testing::AssertionResult AreMatricesEquals(Matrix<float> expected, Matrix<float>actual, float delta)
{      
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
                return ::testing::AssertionFailure() 
                << "The difference at (" << y << "," << x << ") is " << error << ", which exceeds " << delta << ", where" << endl 
                << "expected(y,x) = " << expectedVal << " and" << endl
                << "actual(y,x) = " << actualVal << ".";
            }             
        }
    }
    return ::testing::AssertionSuccess();

}
::testing::AssertionResult  AreMatricesEquals(Matrix<float> a, Matrix<float>b)
{
    return AreMatricesEquals(a, b, 1e-3);
}

TEST_P(MatrixMultiplyTest, SingleElementMatrixTest) {
    initRandom(0);
    auto left = createRandomMatrix(1, 1);
    auto right = createRandomMatrix(1, 1);;

    auto expected = executeMultiplication(referenceImplementation, left, right);
    auto actual = executeMultiplication(currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, SmallSquareMatricesTest) {
    initRandom(12345);
    auto left = createRandomMatrix(5, 5);
    auto right = createRandomMatrix(5, 5);

    auto expected = executeMultiplication(referenceImplementation, left, right);
    auto actual = executeMultiplication(currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TEST_P(MatrixMultiplyTest, MediumRectangularMatricesTest) {
    initRandom(333);
    auto left = createRandomMatrix(30, 50);
    auto right = createRandomMatrix(50, 40);

    auto expected = executeMultiplication(referenceImplementation, left, right);
    auto actual = executeMultiplication(currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

INSTANTIATE_TEST_CASE_P(MultiplePlatforms,
                        MatrixMultiplyTest,
                        ::testing::Values("cuda/cuda", "mpi/mpi-matrix", "smp/smp"));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
