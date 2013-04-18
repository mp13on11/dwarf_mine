#pragma once

#include <iosfwd>
#include <othello_montecarlo/Move.h>
#include <othello_montecarlo/State.h>
#include <vector>
#include <cmath>

void verifyPlayfield(std::vector<Field> expectedField, State& state);

void verifyMoves(const std::vector<Move>& expectedMoves, const std::vector<Move>& actualMoves);

std::ostream& operator<<(std::ostream& out, const std::vector<Move>& moves);
