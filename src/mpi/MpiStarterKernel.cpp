#include "MpiStarterKernel.h"

#include <cstdlib>

using namespace std;

const char* const MpiStarterKernel::ENVIRONMENT_VARIABLE_NAME = "STARTED_WITH_MPIRUN";

bool MpiStarterKernel::wasCorrectlyStarted()
{
    return getenv(ENVIRONMENT_VARIABLE_NAME) != nullptr;
}

MpiStarterKernel::MpiStarterKernel(const shared_ptr<BenchmarkKernel>& kernel) :
        kernel(kernel)
{
}

void MpiStarterKernel::startup(const Arguments&)
{
    // start mpirun somehow
}

void MpiStarterKernel::startup(const vector<string>&)
{
    // should never get called
}

void MpiStarterKernel::run()
{
    // nothing to do here
}

void MpiStarterKernel::shutdown(const string&)
{
    // wait for started process to finish...
}
