#include "MatrixMultiplyTest.h"
#include "Utilities.h"

TYPED_TEST(MatrixMultiplyTest, SingleElementMatrixTest) {
    this->initRandom(0);
    auto left = this->createRandomMatrix(1, 1);
    auto right = this->createRandomMatrix(1, 1);;

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TYPED_TEST(MatrixMultiplyTest, SmallSquareMatricesTest) {
    this->initRandom(12345);
    auto left = this->createRandomMatrix(5, 5);
    auto right = this->createRandomMatrix(5, 5);

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TYPED_TEST(MatrixMultiplyTest, MediumShrinkingRectangularMatricesTest) {
    this->initRandom(333);
    auto left = this->createRandomMatrix(30, 100);
    auto right = this->createRandomMatrix(100, 40);

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}


TYPED_TEST(MatrixMultiplyTest, MediumExpandingRectangularMatricesTest) {
    this->initRandom(4567);
    auto left = this->createRandomMatrix(110, 20);
    auto right = this->createRandomMatrix(20, 130);

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TYPED_TEST(MatrixMultiplyTest, PrimeRectangularMatricesTest) {
    this->initRandom(6543452);
    auto left = this->createRandomMatrix(67, 83);
    auto right = this->createRandomMatrix(83, 109);

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}

TYPED_TEST(MatrixMultiplyTest, BiggerPrimeRectangularMatricesTest) {
    this->initRandom(73653);
    auto left = this->createRandomMatrix(383, 269);
    auto right = this->createRandomMatrix(269, 193);

    auto expected = this->executeMultiplication(*this->referenceImplementation, left, right);
    auto actual = this->executeMultiplication(*this->currentImplementation, left, right);

    EXPECT_TRUE(AreMatricesEquals(expected, actual));
}
