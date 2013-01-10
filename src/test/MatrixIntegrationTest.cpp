#include "Utilities.h"
#include <matrix/MatrixHelper.h>
#include <matrix/Matrix.h>
#include <matrix/MatrixElf.h>
#include <main/ElfFactory.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <fstream>
#include <future>
#include <sys/wait.h>
#include <signal.h>

using namespace std;

TEST(MatrixIntegrationTest, TestItNowwwwww)
{
    const int TIMEOUT_SECONDS = 10;
    pid_t pid = fork();
    if(pid == 0) // child process
    {
        char* argv[] = {(const char*)"/usr/bin/mpirun", 
                        (const char*)"-n", "8",
                        (const char*)"--tag-output",
                        (const char*)"build/src/main/dwarf_mine", 
                        (const char*)"-m", "smp", 
                        (const char*)"-n", "1",
                        (const char*)"-w", "0",
                        (const char*)"-q",
                        (const char*)"-i", "small_input.bin",
                        (const char*)"-o", "small_output.bin",
                        (const char*)"--import_configuration", "test_config.cfg",
                        nullptr};
        execv(argv[0], argv);
        exit(-1);
    }

    auto future = async(std::launch::async, [pid]()->bool
    {
        int status;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) && (WEXITSTATUS(status) == 0);
    });

    auto status = future.wait_for(std::chrono::seconds(TIMEOUT_SECONDS));

    if(status != future_status::ready)
    {
        kill(pid, SIGKILL);
        future.wait();
        FAIL() << "Process timed out";
    }

    ASSERT_TRUE(future.get()) << "Process not exited normally";

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