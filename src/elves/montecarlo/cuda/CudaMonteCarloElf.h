#pragma once

#include <MonteCarloElf.h>
#include <OthelloUtil.h>
#include <OthelloState.h>

class CudaMonteCarloElf : public MonteCarloElf
{
public:
	virtual OthelloResult getBestMoveFor(OthelloState& state, size_t reiterations);
	
};
