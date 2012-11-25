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
    static const std::size_t BLOCK_ROWS;
    static const std::size_t BLOCK_COLUMNS;

    const int rank;
    const int groupSize;
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;
    Matrix<float> temp;

    void broadcastSizes();
    void scatterMatrices();
    void multiply();
    void gatherResult();
    std::vector<float> distributeBuffer(const float*, const std::size_t rows, const std::size_t columns);
    std::vector<float> transposeMatrix(const float*, const std::size_t rows, const std::size_t columns);
    bool const isRoot() const;
};

inline bool const MpiMatrixKernel::isRoot() const
{
    return rank == ROOT_RANK;
}

inline std::size_t MpiMatrixKernel::requiredInputs() const
{
    return 2;
}
