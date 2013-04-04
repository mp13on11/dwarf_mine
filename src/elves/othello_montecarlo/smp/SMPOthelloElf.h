#pragma once

#include <OthelloElf.h>
#include "OthelloUtil.h"
#include <chrono>
#include <vector>

class SMPOthelloElf : public OthelloElf
{
public:
    virtual std::vector<Result> getMovesFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
    std::vector<RandomGenerator> _generators;
    std::chrono::high_resolution_clock::time_point _end;
};

