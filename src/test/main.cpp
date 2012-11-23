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




class MatrixMultiplyTest : public ::testing::Test
{
protected:

    virtual void SetUp() {
        MatrixHelper::writeMatrixTo(matrixAFile, createTempMatrix(5,5));
        MatrixHelper::writeMatrixTo(matrixBFile, createTempMatrix(5,5));
        ofstream file;
        file.open(matrixCFile);
    }

    virtual void TearDown() {
        remove(matrixAFile.c_str());
        remove(matrixBFile.c_str());
        remove(matrixCFile.c_str());
    }

    Matrix<float> createTempMatrix(size_t rows, size_t columns)
    {
        uniform_real_distribution<float> distribution(-100, +100);
        mt19937 engine(12345);
        auto generator = bind(distribution, engine);
        Matrix<float> m(rows, columns);

        for(size_t y = 0; y<rows; y++)
        {
            for(size_t x = 0; x<columns; x++)
            {
                m(y, x) = generator();
            }
        }
        return m;
    }

    string matrixAFile = "a.txt";
    string matrixBFile = "b.txt";
    string matrixCFile = "c.txt";
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
            float error = fabs(expected(y,x) - actual(y,x));
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
    return AreMatricesEquals(a, b, 1e-1);
}

void startProcess(initializer_list<string> args)
{
    system(boost::algorithm::join(args, " ").c_str());
}

TEST_F(MatrixMultiplyTest, GoldTest) { 
    startProcess({"src/gold/gold", "a.txt", "b.txt", "c.txt"});
    startProcess({"src/cuda/cuda", "a.txt", "b.txt", "c2.txt"});
    auto expected = MatrixHelper::readMatrixFrom("c.txt");
    auto actual = MatrixHelper::readMatrixFrom("c2.txt");
    EXPECT_TRUE(AreMatricesEquals(expected, actual));
    //EXPECT_NEAR (182.0, square_root (324.0), 1e-4);
    //EXPECT_NEAR (25.4, square_root (645.16), 1e-4);
    //EXPECT_NEAR (50.3321, square_root (2533.310224), 1e-4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}