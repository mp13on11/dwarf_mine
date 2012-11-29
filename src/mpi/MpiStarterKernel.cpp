#include "MpiStarterKernel.h"

#include <cstdio>
#include <cstdlib>

using namespace std;

const char* const MpiStarterKernel::ENVIRONMENT_VARIABLE_NAME = "STARTED_WITH_MPIRUN";

bool MpiStarterKernel::wasStartedCorrectly()
{
    return getenv(ENVIRONMENT_VARIABLE_NAME) != nullptr;
}

MpiStarterKernel::MpiStarterKernel(const shared_ptr<BenchmarkKernel>& kernel) :
        kernel(kernel)
{
}

void MpiStarterKernel::startup(const Arguments& arguments)
{
    this->arguments = arguments;
}

void MpiStarterKernel::startup(const vector<string>&)
{
    // should never get called
}

void MpiStarterKernel::run()
{
    int status = setenv(ENVIRONMENT_VARIABLE_NAME, "true", 0);

    if (status != 0)
    {
        perror("setenv");
        return;
    }

    string command = "mpirun -n 4 " + arguments.toString();
    status = system(command.c_str());

    if (status == -1)
        perror("Could not start mpirun");
}

void MpiStarterKernel::shutdown(const string&)
{
    // wait for started process to finish...
}
