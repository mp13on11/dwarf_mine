#include "benchmark/BenchmarkKernel.h"

#include <iostream>

using namespace std;

BenchmarkKernel::~BenchmarkKernel()
{
}

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << "<input files>... <output file>" << endl;
        return 1;
    }

    vector<string> inputs(argv + 1, argv + argc - 1);
    string output(argv[argc - 1]);

    auto kernel = createKernel();
    kernel->startup(inputs);
    kernel->run();
    kernel->shutdown(output);
}
