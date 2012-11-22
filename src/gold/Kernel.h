#include "../benchmark-lib/BenchmarkKernel.h"

class Kernel : public BenchmarkKernel
{
public:
	void startup(const std::string& inputFilename) override;
	void run() override;
	void shutdown(const std::string& outputFilename) override;
};