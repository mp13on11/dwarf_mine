#include "benchmark/Arguments.h"
#include "benchmark/BenchmarkKernel.h"
#include "benchmark/InvalidCommandLineException.h"

#include <chrono>
#include <iostream>
#include <functional>
#include <map>

using namespace std;

BenchmarkKernel::~BenchmarkKernel()
{
}

map<string, float> benchmark(function<void()> func, size_t iterations = 1)
{
    typedef chrono::high_resolution_clock clock;
    typedef chrono::microseconds microseconds;
    clock::time_point before = clock::now();

    for(size_t i=0; i<iterations; i++)
    {
        func();
    }

    clock::time_point after = clock::now();
    microseconds total_ms = (after - before);

    return map<string,float>{{"time (µs)", total_ms.count()}};
}

int main(int argc, const char* argv[])
{
    Arguments args;

    try
    {
        args = Arguments(argc, argv);
    }
    catch (InvalidCommandLineException& e)
    {
        e.what();
        Arguments::printUsage(argv[0], cerr);
        return 1;
    }

    auto kernel = createKernel();

    // check for required number of input arguments
    if (args.inputFileNames().size() < kernel->requiredInputs())
    {
        cerr << argv[0] << " requires " << kernel->requiredInputs() << " input files..." << endl;
        Arguments::printUsage(argv[0], cerr);
        return 1;
    }

    // execute kernel
    kernel->startup(args.inputFileNames());

    auto stats = benchmark([&](){kernel->run();}, args.iterations());

    kernel->shutdown(args.outputFileName());

    if (!kernel->statsShouldBePrinted())
        return 0;

    // print statistics
    for (auto& kv : stats) {
        cout << kv.first << ": " << kv.second << std::endl;
    }

    return 0;
}
