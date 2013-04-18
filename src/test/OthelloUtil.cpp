/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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
