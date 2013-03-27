#pragma once

#include <MonteCarloElf.h>
#include <OthelloUtil.h>
#include <OthelloState.h>

class CudaMonteCarloElf : public MonteCarloElf
{
public:
    virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
private:
	OthelloResult getBestMoveForSimple(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
	OthelloResult getBestMoveForStreamed(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed);
};
