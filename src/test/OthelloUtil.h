#pragma once

#include <iosfwd>
#include <montecarlo/OthelloMove.h>
#include <montecarlo/OthelloState.h>
#include <vector>
#include <cmath>

using namespace std;

void verifyPlayfield(std::vector<Field> expectedField, OthelloState& state);

void verifyMoves(const std::vector<OthelloMove>& expectedMoves, const std::vector<OthelloMove>& actualMoves);