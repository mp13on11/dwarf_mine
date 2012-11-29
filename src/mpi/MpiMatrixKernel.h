#pragma once

#include "benchmark/BenchmarkKernel.h"
#include "tools/Matrix.h"


class MpiMatrixKernel : public BenchmarkKernel
{
public:
    MpiMatrixKernel();
    MpiMatrixKernel(const MpiMatrixKernel &copy) = delete;
    MpiMatrixKernel& operator=(const MpiMatrixKernel &rhs) = delete;
    virtual ~MpiMatrixKernel();

    virtual std::size_t requiredInputs() const;
    virtual void startup(const std::vector<std::string>& arguments);
    virtual void run();
    virtual void shutdown(const std::string& outputFilename);
    virtual bool statsShouldBePrinted() const;

private:
    static const int ROOT_RANK;
    static const std::size_t BLOCK_SIZE;

    const int rank;
    const int groupSize;
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;
    float *rowBuffer;
    float *columnBuffer;
    float *resultBuffer;
    std::size_t fullRows;
    std::size_t fullColumns;
    std::size_t sentRows;
    std::size_t sentColumns;

    void broadcastSizes();
    void scatterMatrices(std::size_t round);
    void multiply(std::size_t round);
    void gatherResult(std::size_t round);
    bool isRoot() const;

    std::size_t blockIndex(std::size_t round, int rank) const;
    std::size_t rowIndex(std::size_t round, int rank) const;
    std::size_t columnIndex(std::size_t round, int rank) const;

    std::size_t blocksPerRow() const;
    std::size_t blocksPerColumn() const;
    std::size_t blockCount() const;
};

inline bool MpiMatrixKernel::statsShouldBePrinted() const
{
    return isRoot();
}

inline bool MpiMatrixKernel::isRoot() const
{
    return rank == ROOT_RANK;
}

inline std::size_t MpiMatrixKernel::requiredInputs() const
{
    return 2;
}

inline std::size_t MpiMatrixKernel::rowIndex(std::size_t round, int rank) const
{
    return sentRows * (blockIndex(round, rank) / blocksPerRow());
}

inline std::size_t MpiMatrixKernel::columnIndex(std::size_t round, int rank) const
{
    return sentColumns * (blockIndex(round, rank) % blocksPerRow());
}

inline std::size_t MpiMatrixKernel::blockIndex(std::size_t round, int rank) const
{
    return (round * groupSize) + rank;
}

inline std::size_t MpiMatrixKernel::blockCount() const
{
    return blocksPerRow() * blocksPerColumn();
}

inline std::size_t MpiMatrixKernel::blocksPerRow() const
{
    std::size_t result = fullColumns / sentColumns;
    if (fullColumns % sentColumns != 0)
        result += 1;

    return result;
}

inline std::size_t MpiMatrixKernel::blocksPerColumn() const
{
    std::size_t result = fullRows / sentRows;
    if (fullRows % sentRows != 0)
        result += 1;

    return result;
}
