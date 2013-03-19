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
    MatrixScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);
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

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();
    virtual void calculateOnMaster(const MatrixSlice& definition, Matrix<float>& result) const;
    virtual Matrix<float> dispatchAndReceive() const;
    virtual void collectResults(const std::vector<MatrixSlice>& slices, Matrix<float>& result) const;
    std::pair<Matrix<float>, Matrix<float>> sliceMatrices(const MatrixSlice& definition) const;

private:
    const MatrixSlice* distributeSlices(const std::vector<MatrixSlice>& slices) const;
};
