#pragma once

#include "Matrix.h"
#include "common/SchedulerTemplate.h"

#include <functional>
#include <vector>

class MatrixElf;
class MatrixSlice;

class MatrixScheduler: public SchedulerTemplate<MatrixElf>
{
public:
    MatrixScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixScheduler();

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);

protected:
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    virtual bool hasData() const;
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual void doBenchmarkDispatch(NodeId node);

private:
    void calculateOnSlave();
    void orchestrateCalculation();
    Matrix<float> dispatchAndReceive() const;
    const MatrixSlice* distributeSlices(const std::vector<MatrixSlice>& slices) const;
    void calculateOnMaster(const MatrixSlice& definition, Matrix<float>& result) const;
    void collectResults(const std::vector<MatrixSlice>& slices, Matrix<float>& result) const;
    std::pair<Matrix<float>, Matrix<float>> sliceMatrices(const MatrixSlice& definition) const;
};
