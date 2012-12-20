#include "MatrixMultiplyTest.h"
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
