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

    OthelloState _state;
    OthelloResult _result;
    size_t _repetitions;
    size_t _localRepetitions;
    size_t _commonSeed;

private:
    void orchestrateCalculation();
    void calculateOnSlave();
    void calculate();
    void distributeInput();
    size_t distributeCommonParameters();
    void distributePlayfield(size_t size);
    void collectResults();
    std::vector<OthelloResult> gatherResults();
};
