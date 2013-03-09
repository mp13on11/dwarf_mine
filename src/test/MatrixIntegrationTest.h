#pragma once

#include "matrix/Matrix.h"

#include <gtest/gtest.h>
#include <cstdlib>

class MatrixIntegrationTest : public testing::Test
{
public:
    static void executeWith(const char* const matrixCategory);
    static void setupConfigFile();
    static pid_t spawnChildProcess(const char* const matrixCategory);
    static std::tuple<Matrix<float>, Matrix<float>> readMatrices();

protected:
    virtual void TearDown();
};
