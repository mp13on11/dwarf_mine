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
#include <boost/lexical_cast.hpp>
#include <sys/wait.h>
#include <signal.h>

using namespace std;

const int           TIMEOUT_SECONDS = 10;
const int           NUM_NODES       = 8;
const char* const   CONF_FILENAME   = "test_config.cfg";
const char* const   INPUT_FILENAME  = "small_input.bin";
const char* const   OUTPUT_FILENAME = "small_output.bin";

void setupConfigFile()
{
    ofstream config(CONF_FILENAME);
    for (int i=0; i<NUM_NODES; ++i)
        config << i << " " << 1 << endl;
}

pid_t spawnChildProcess()
{
    pid_t pid = fork();
    if(pid == 0) // child process
    {
        execl("/usr/bin/mpirun",
            "/usr/bin/mpirun", 
            "-n", boost::lexical_cast<string>(NUM_NODES).c_str(),
            "--tag-output",
            "build/src/main/dwarf_mine", 
            "-m", "smp", 
            "-n", "1",
            "-w", "0",
            "-q",
            "-i", INPUT_FILENAME,
            "-o", OUTPUT_FILENAME,
            "--import_configuration", CONF_FILENAME,
            nullptr
        );
        exit(-1);
    }
    return pid;
}

TEST(MatrixIntegrationTest, TestSmallInputSMPScheduling)
{
    setupConfigFile();
    pid_t pid = spawnChildProcess();

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

    remove(OUTPUT_FILENAME);
    remove(CONF_FILENAME);
}
