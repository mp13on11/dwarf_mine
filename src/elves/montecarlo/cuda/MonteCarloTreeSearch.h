#pragma once

#include <OthelloField.h>

extern void compute(size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits);
extern void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom);