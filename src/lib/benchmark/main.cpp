#include "benchmark/BenchmarkKernel.h"

BenchmarkKernel::~BenchmarkKernel()
{
}

int main(int argc, const char *argv[])
{
    auto kernel = createKernel();
    kernel->startup(std::vector<std::string>{"../a.txt", "../b.txt"});
    kernel->run();
    kernel->shutdown("../c.txt");
}