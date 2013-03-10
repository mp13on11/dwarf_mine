#pragma once

#include "matrix/Matrix.h"

#include <gtest/gtest.h>
#include <cstdlib>

class MatrixIntegrationTest : public testing::Test
{
protected:
    static void executeWith(
        const char* matrixCategory,
        const char* scheduling = "");
    static void setupConfigFile();
    static pid_t spawnChildProcess(
        const char* matrixCategory,
        const char* scheduling = "");
    static std::tuple<Matrix<float>, Matrix<float>> readMatrices();
    virtual void TearDown();
};
