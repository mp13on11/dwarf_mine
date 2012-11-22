#include "benchmark/BenchmarkKernel.h"
#include "tools/Matrix.h"

class Kernel : public BenchmarkKernel
{
private:
    Matrix<float> a,b,c;

public:
    void startup(const std::vector<std::string>& arguments);
    void run();
    void shutdown(const std::string& outputFilename);
};