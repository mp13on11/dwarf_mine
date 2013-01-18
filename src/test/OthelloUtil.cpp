#include "OthelloUtil.h"

#include <iostream>
#include <montecarlo/OthelloMove.h>
#include <vector>
#include <cmath>
#include <gtest/gtest.h>

using namespace std;



void verifyPlayfield(vector<Field> expectedField, OthelloState& state)
{
    int sideLength = sqrt(expectedField.size());
    ASSERT_EQ(sideLength, state.playfieldSideLength());
    for (int i = 0; i < sideLength; ++i)
    {
        for (int j = 0; j < sideLength; ++j)
        {
            ASSERT_EQ(expectedField[j*sideLength + i], state.atPosition(i, j));
        }
    }
}


void verifyMoves(const vector<OthelloMove>& expectedMoves, const vector<OthelloMove>& actualMoves)
{    
    vector<OthelloMove> matchedMoves;
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