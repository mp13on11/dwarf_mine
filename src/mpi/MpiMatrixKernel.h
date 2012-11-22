#pragma once

#include "benchmark/BenchmarkKernel.h"
#include "tools/Matrix.h"

class MpiMatrixKernel : public BenchmarkKernel
{
public:
    MpiMatrixKernel();
    virtual std::size_t requiredInputs() const;
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);

private:
    static const int ROOT_RANK;
    static const int BLOCK_SIZE;

    const int rank;
    const int groupSize;
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    void broadcastSizes();
    void scatterMatrices();
    void multiply();
    void gatherResult();
};

inline std::size_t MpiMatrixKernel::requiredInputs() const
{
    return 2;
}
