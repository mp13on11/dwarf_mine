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

std::vector<const char*> getPlatforms()
{
    return {
#ifdef HAVE_CUDA
        "cuda",
#endif
        "smp"
    };
}

INSTANTIATE_TEST_CASE_P(
    MultiplePlatforms,
    MatrixMultiplyTest,
    testing::ValuesIn(getPlatforms()));

size_t blockSize = 24;


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
