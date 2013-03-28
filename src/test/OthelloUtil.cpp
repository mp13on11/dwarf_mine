#include "OthelloUtil.h"

#include <iostream>
#include <othello_montecarlo/Move.h>
#include <vector>
#include <cmath>
#include <gtest/gtest.h>

using namespace std;



void verifyPlayfield(vector<Field> expectedField, State& state)
{
    int sideLength = sqrt(expectedField.size());
    ASSERT_EQ(sideLength, state.playfieldSideLength());
    for (int i = 0; i < sideLength; ++i)
    {
        for (int j = 0; j < sideLength; ++j)
        {
            ASSERT_EQ(expectedField[j*sideLength + i], state.playfield(i, j));
        }
    }
}


void verifyMoves(const vector<Move>& expectedMoves, const vector<Move>& actualMoves)
{    
    vector<Move> matchedMoves;
    ASSERT_EQ(expectedMoves.size(), actualMoves.size());
    for (const auto& actualMove : actualMoves)
    {
        for (auto j = expectedMoves.begin(); j != expectedMoves.end(); ++j)
        {
            if (actualMove.x == (*j).x
                && actualMove.y == (*j).y)
            {
                matchedMoves.push_back(*j);
            }
        }
    }
    ASSERT_EQ(expectedMoves.size(), matchedMoves.size());
}


ostream& operator<<(ostream& out, const vector<Move>& moves)
{
    for ( const auto& move : moves)
    {
        out << "{"<<move.x<<", "<<move.y<<"} ";
    }
    return out;
}
