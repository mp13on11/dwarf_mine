#include "benchmark/BenchmarkKernel.h"

#include <iostream>
#include <chrono>
#include <functional>
#include <map>
#include <boost/lexical_cast.hpp>

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
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <options> <input files>... <output file>" << endl;
        return 1;
    }

    // split args into first as inputs and the last as output
    vector<string> inputs(argv + 1, argv + argc - 1);
    string output(argv[argc - 1]);

    int iterations = 1;
    if(inputs[0] == "--iterations")
    {
        iterations = boost::lexical_cast<int>(inputs[1]);
        inputs.erase(inputs.begin());
        inputs.erase(inputs.begin());
    }

    auto kernel = createKernel();

    // check for required number of input arguments
    if (inputs.size() < kernel->requiredInputs())
    {
        cerr << argv[0] << " requires " << kernel->requiredInputs() << " input files..." << endl;
        return 1;
    }

    // execute kernel
    kernel->startup(inputs);

    auto stats = benchmark([&](){kernel->run();}, iterations);

    kernel->shutdown(output);

    // print statistics
    for (auto& kv : stats) {
        cout << kv.first << ": " << kv.second << std::endl;
    }
}
