#include "BenchmarkKernel.h"

BenchmarkKernel::~BenchmarkKernel()
{
}

int main(int argc, const char *argv[])
{
	auto kernel = createKernel();
	kernel->startup("sample-input");
	kernel->run();
	kernel->shutdown("sample-output");
}