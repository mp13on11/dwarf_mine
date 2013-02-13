#pragma once

#include <OthelloField.h>
#include <OthelloResult.h>

extern void leafSimulation(size_t reiterations, size_t dimension, Field* playfield, Player currentPlayer, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits);
extern void gameSimulation(size_t reiterations, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results);

extern void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom);
extern void testByLeafSimulation(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);