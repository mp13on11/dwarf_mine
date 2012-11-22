#include "benchmark/BenchmarkKernel.h"
#include "tools/Matrix.h"

class Kernel : public BenchmarkKernel
{
private:
    Matrix<float> a,b,c;

public:
    std::size_t requiredInputs() const;
    void startup(const std::vector<std::string>& arguments);
    void run();
    void shutdown(const std::string& outputFilename);
};

inline std::size_t Kernel::requiredInputs() const
{
    return 2;
}
