#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"

#include <functional>
#include <vector>

class MatrixElf;
class MatrixSlice;

class MatrixOnlineScheduler: public MatrixScheduler
{
public:
    MatrixOnlineScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixOnlineScheduler();

    virtual void generateData(const DataGenerationParameters& params);

protected:
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    virtual void doSimpleDispatch();

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();
    virtual void calculateOnMaster(const MatrixSlice& definition, Matrix<float>& result) const;
    virtual Matrix<float> dispatchAndReceive() const;
    virtual void collectResults(const std::vector<MatrixSlice>& slices, Matrix<float>& result) const;
};
