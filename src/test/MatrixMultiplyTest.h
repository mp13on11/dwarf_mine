#pragma once

#include <gtest/gtest.h>
#include <matrix/MatrixElf.h>
#include <cstdlib>
#include <functional>
#include <memory>

class MatrixMultiplyTest : public ::testing::TestWithParam<const char*>
{
protected:
    //
    // Members
    virtual void SetUp();
    virtual void TearDown();

    Matrix<float> createRandomMatrix(size_t rows, size_t columns);
    void initRandom(uint seed);
    Matrix<float> executeMultiplication(MatrixElf& elf, const Matrix<float>& a, const Matrix<float>& b);
    void startProcess(std::initializer_list<std::string> args);

    //
    // Instance variables
    std::function<float()> generator;

    std::string inputAFile;
    std::string inputBFile;
    std::string outputFile;

    std::unique_ptr<MatrixElf> referenceImplementation;
    std::unique_ptr<MatrixElf> currentImplementation;
};
