#pragma once

#include "common/SchedulerTemplate.h"
#include "MonteCarloElf.h"
#include <functional>
#include <vector>
#include <mpi.h>

class MonteCarloElf;
class OthelloResult;

class MonteCarloScheduler: public SchedulerTemplate<MonteCarloElf>
{
public:
    MonteCarloScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MonteCarloScheduler() = default;

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);


protected:
    virtual bool hasData() const;
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual void doBenchmarkDispatch(NodeId node);

    OthelloState _state;
    OthelloResult _result;
    size_t _repetitions;
    size_t _localRepetitions;

private:
    void orchestrateCalculation();
    void calculateOnSlave();
    void calculate();
    void distributeInput();
    void collectInput();
    void collectResults();
    std::vector<OthelloResult> gatherResults();
};
