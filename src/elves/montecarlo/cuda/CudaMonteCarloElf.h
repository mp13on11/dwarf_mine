#pragma once

#include <MonteCarloElf.h>
#include <OthelloUtil.h>
#include <OthelloState.h>

class CudaMonteCarloElf : public MonteCarloElf
{
public:
    virtual std::vector<OthelloResult> getMovesFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
};
