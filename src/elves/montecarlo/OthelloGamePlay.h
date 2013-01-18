#pragma once

#include <OthelloUtil.h>
#include <OthelloState.h>

class OthelloGamePlay
{
public:
	OthelloResult getBestMoveFor(OthelloState& state, size_t iterations);
};