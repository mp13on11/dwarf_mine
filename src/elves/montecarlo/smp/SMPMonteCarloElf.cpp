#include "SMPMonteCarloElf.h"
#include <OthelloGamePlay.h>


OthelloResult SMPMonteCarloElf::calculateBestMove(const OthelloState& state)
{
	OthelloGamePlay game;
    OthelloState copy = state;
    return game.getBestMoveFor(copy, 100U);
}