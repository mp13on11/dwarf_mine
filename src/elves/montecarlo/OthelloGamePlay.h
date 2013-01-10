#pragma once

#include <OthelloMove.h>
#include <OthelloState.h>

class OthelloGamePlay
{
	OthelloMove getBestMoveFor(OthelloState& state, size_t iterations);
};