#include "benchmark/BenchmarkKernel.h"

BenchmarkKernel::~BenchmarkKernel()
{
}

int main(int argc, const char *argv[])
{
	auto kernel = createKernel();
	kernel->startup(std::vector<std::string>{"sample-input"});
	kernel->run();
	kernel->shutdown("sample-output");
}
