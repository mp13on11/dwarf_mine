#pragma once

#include <MonteCarloElf.h>
#include "OthelloUtil.h"
#include <chrono>
#include <vector>

class SMPMonteCarloElf : public MonteCarloElf
{
public:
    virtual std::vector<OthelloResult> getMovesFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
    std::vector<RandomGenerator> _generators;
    std::chrono::high_resolution_clock::time_point _end;
};

