#include "../benchmark-lib/BenchmarkKernel.h"

class Kernel : public BenchmarkKernel
{
public:
	void startup(const std::vector<std::string>& arguments);
	void run();
	void shutdown(const std::string& outputFilename);
};