#ifndef BENCHMARK_KERNEL_H
#define BENCHMARK_KERNEL_H

class BenchmarkKernel
{
public:
	virtual ~BenchmarkKernel();
	virtual void startup(const std::string& inputFilename) = 0;
	virtual void run() = 0;
	virtual void shutdown(const std::string& outputFilename) = 0;
};

inline BenchmarkKernel::~BenchmarkKernel()
{
}

#endif /* BENCHMARK_KERNEL_H */
