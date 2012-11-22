#ifndef BENCHMARK_KERNEL_H
#define BENCHMARK_KERNEL_H

using namespace std;

class BenchmarkKernel
{
public:
	virtual startup(const string& inputFilename) = 0;
	virtual run() = 0;
	virtual shutdown(const string& outputFilename) = 0;
}

#endif /* BENCHMARK_KERNEL_H */
