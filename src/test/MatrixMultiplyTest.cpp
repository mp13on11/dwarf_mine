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
