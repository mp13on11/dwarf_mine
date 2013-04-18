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

#include "MatrixIntegrationTest.h"
#include "Utilities.h"
#include "matrix/MatrixHelper.h"
#include "matrix/MatrixElf.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "common/SchedulerFactory.h"

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
const char* const   INPUT_FILENAME  = "example_data/matrix/small_input.bin";
const char* const   OUTPUT_FILENAME = "small_output.bin";
const char* const   MPIRUN_PATH     = MPIEXEC; // defined by CMake file
const char* const   EXECUTABLE_PATH = "build/src/main/dwarf_mine";

TEST_F(MatrixIntegrationTest, TestSmallInputSMPSchedulingOffline)
{
    MatrixIntegrationTest::executeWith("matrix");
}

TEST_F(MatrixIntegrationTest, TestSmallInputSMPSchedulingOnline)
{
    MatrixIntegrationTest::executeWith("matrix_online", "row-wise");
}

void MatrixIntegrationTest::executeWith(
    const char* matrixCategory,
    const char* scheduling)
{
    MatrixIntegrationTest::setupConfigFile();
    pid_t pid = MatrixIntegrationTest::spawnChildProcess(matrixCategory, scheduling);

    auto future = async(std::launch::async, [pid]() -> bool
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

    Matrix<float> expectedMatrix, actualMatrix;
    std::tie(expectedMatrix, actualMatrix) = MatrixIntegrationTest::readMatrices();

    EXPECT_TRUE(AreMatricesEquals(expectedMatrix, actualMatrix));
}

void MatrixIntegrationTest::TearDown()
{
    remove(OUTPUT_FILENAME);
    remove(CONF_FILENAME);
}

void MatrixIntegrationTest::setupConfigFile()
{
    ofstream config(CONF_FILENAME);
    for (int i=0; i<NUM_NODES; ++i)
        config << 1.0/NUM_NODES << endl;
}

pid_t MatrixIntegrationTest::spawnChildProcess(
    const char* matrixCategory,
    const char* scheduling)
{
    string schedulingStrategy(scheduling);
    pid_t pid = fork();
    if(pid == 0) // child process
    {
        bool usesOnlineScheduling = schedulingStrategy != "";
        execl(MPIRUN_PATH,
            MPIRUN_PATH,
            "-n", boost::lexical_cast<string>(NUM_NODES).c_str(),
            "--tag-output",
            EXECUTABLE_PATH,
            "-m", "smp",
            "-n", "1",
            "-w", "0",
            "-q",
            "-i", INPUT_FILENAME,
            "-o", OUTPUT_FILENAME,
            "--import_configuration", CONF_FILENAME,
            "-c", matrixCategory,
            (usesOnlineScheduling ? "--mpi_thread_multiple" : ""),
            (schedulingStrategy != "" ? "-s" : ""),
            (schedulingStrategy != "" ? scheduling : ""),
            nullptr
        );
        exit(-1);
    }
    return pid;
}

std::tuple<Matrix<float>, Matrix<float>> MatrixIntegrationTest::readMatrices()
{
    ifstream input(INPUT_FILENAME, ios_base::binary);
    auto inputMatrices = MatrixHelper::readMatrixPairFrom(input);
    auto actualMatrix = MatrixHelper::readMatrixFrom(OUTPUT_FILENAME);

    auto leftMatrix = inputMatrices.first;
    auto rightMatrix = inputMatrices.second;

    SMPMatrixElf elf;
    auto expectedMatrix = elf.multiply(leftMatrix, rightMatrix);
    return make_tuple<Matrix<float>, Matrix<float>>(move(expectedMatrix), move(actualMatrix));
