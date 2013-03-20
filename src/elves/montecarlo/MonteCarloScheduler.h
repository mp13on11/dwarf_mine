#pragma once

#include "common/SchedulerTemplate.h"
#include "MonteCarloElf.h"
#include <functional>
#include <vector>

class MonteCarloElf;
struct OthelloResult;

class MonteCarloScheduler: public SchedulerTemplate<MonteCarloElf>
{
public:
    MonteCarloScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);
    virtual ~MonteCarloScheduler() = default;

    virtual void provideData(std::istream& input);
    virtual void outputData(std::ostream& output);
    virtual void generateData(const DataGenerationParameters& params);


protected:
    virtual bool hasData() const;
    virtual void doDispatch();
    virtual void doSimpleDispatch();
    virtual void doBenchmarkDispatch(int node);

    OthelloState _state;
    OthelloResult _result;
    size_t _repetitions;
    size_t _localRepetitions;
    size_t _commonSeed;

private:
    void doDispatch(BenchmarkResult nodeSet);
    void orchestrateCalculation();
    void calculateOnSlave();
    void calculate();
    void distributeInput(BenchmarkResult nodeSet);
    void collectInput(BenchmarkResult nodeSet);
    void collectResults(BenchmarkResult nodeSet);
    std::vector<OthelloResult> gatherResults(BenchmarkResult nodeSet);
};
