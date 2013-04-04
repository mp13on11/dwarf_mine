#pragma once

#include <MonteCarloElf.h>
#include <OthelloUtil.h>
#include <OthelloState.h>

class CudaMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
    OthelloResult getBestMoveForSingleStream(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
    OthelloResult getBestMoveForMultipleStream(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
};
