#pragma once

#include <OthelloField.h>
#include <OthelloResult.h>
#include <vector>

extern void leafSimulation(size_t reiterations, size_t dimension, Field* playfield, Player currentPlayer, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits);
extern void gameSimulation(size_t numberOfBlocks, size_t iterations, size_t* seeds, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results);

extern void testRandomNumberProxy(float fakedRandom, size_t maximum, size_t* randomMoveIndex);
extern void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom);
extern void testSimulateGameLeafProxy(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);