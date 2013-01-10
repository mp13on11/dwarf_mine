#include "Utilities.h"
#include <matrix/MatrixHelper.h>
#include <matrix/Matrix.h>
#include <matrix/MatrixElf.h>
#include <main/ElfFactory.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <fstream>

using namespace std;

TEST(MatrixIntegrationTest, TestItNowwwwww)
{
    system("mpirun -n 8 --tag-output build/src/main/dwarf_mine -m smp -n 1 -w 0 -i" 
        "small_input.bin -o small_output.bin --import_configuration test_config.cfg");

    ifstream input("small_input.bin", ios_base::binary);
    auto inputMatrices = MatrixHelper::readMatrixPairFrom(input);
    auto actualMatrix = MatrixHelper::readMatrixFrom("small_output.bin");

    auto leftMatrix = inputMatrices.first;
    auto rightMatrix = inputMatrices.second;

    auto elf = createElfFactory("smp", "matrix")->createElf();
    auto expectedMatrix = static_cast<MatrixElf*>(elf.get())->multiply(leftMatrix, rightMatrix);

    EXPECT_TRUE(AreMatricesEquals(expectedMatrix, actualMatrix));

    remove("small_output.bin");
}