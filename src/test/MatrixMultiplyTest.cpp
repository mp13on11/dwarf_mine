#include "MatrixMultiplyTest.h"
#include "Utilities.h"
#include "GoldMatrixElf.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <main/ElfFactory.h>

#include <random>
#include <sstream>
#include <boost/algorithm/string/join.hpp>

using namespace std;

void MatrixMultiplyTest::SetUp() {
    inputBFile = "b.txt";
    outputFile = "c.txt";

    auto factory = createElfFactory(GetParam(), "matrix");
    auto elf = factory->createElf();

    currentImplementation.reset(static_cast<MatrixElf*>(elf.release()));
    referenceImplementation.reset(new GoldMatrixElf());
}

void MatrixMultiplyTest::TearDown() {
    remove(inputAFile.c_str());
    remove(inputBFile.c_str());
    remove(outputFile.c_str());
}

Matrix<float> MatrixMultiplyTest::createRandomMatrix(size_t rows, size_t columns)
{
    Matrix<float> m(rows, columns);
    MatrixHelper::fill(m, generator);
    return m;
}

void MatrixMultiplyTest::initRandom(uint seed)
{
    auto distribution = uniform_real_distribution<float>(-100, +100);
    auto engine = mt19937(seed);
    generator = bind(distribution, engine);
}

Matrix<float> MatrixMultiplyTest::executeMultiplication(MatrixElf& elf, const Matrix<float>& a, const Matrix<float>& b)
{
    return elf.multiply(a, b);
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
