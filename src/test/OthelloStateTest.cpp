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


#include "OthelloStateTest.h"
#include "OthelloUtil.h"
#include <othello_montecarlo/Move.h>
#include <othello_montecarlo/Exceptions.h>
#include <iostream>
#include <cmath>

using namespace std;

void OthelloStateTest::SetUp()
{
    state = new State(8);
}

void OthelloStateTest::TearDown()
{
    delete state;
}

TEST_F(OthelloStateTest, InitializationTest)
{
    ASSERT_EQ(Player::White, state->getCurrentPlayer());
    ASSERT_EQ(Player::Black, state->getCurrentEnemy());
}

TEST_F(OthelloStateTest, CreationTest)
{
    ASSERT_THROW(State(9), InvalidFieldSizeException);
}

TEST_F(OthelloStateTest, DoMoveThrowOnOutOfBoundsTest)
{
    ASSERT_THROW(state->doMove(Move{8, 8}), InvalidMoveException);
    ASSERT_THROW(state->doMove(Move{-1, -1}), InvalidMoveException);
    ASSERT_THROW(state->doMove(Move{5, 8}), InvalidMoveException);
    ASSERT_THROW(state->doMove(Move{3, -5}), InvalidMoveException);   
}

TEST_F(OthelloStateTest, DoMoveThrowOnAccessOccupiedFieldTest)
{
    ASSERT_THROW(state->doMove(Move{3, 3}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(Move{3, 4}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(Move{4, 3}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(Move{4, 4}), OccupiedFieldException);
}


TEST_F(OthelloStateTest, GetPossibleMovesTest)
{
    auto actualMoves = state->getPossibleMoves();
    vector<Move> expectedMoves{
        Move{4, 2}, 
        Move{5, 3}, 
        Move{2, 4},
        Move{3, 5}
    };
    verifyMoves(expectedMoves, actualMoves);
}



TEST_F(OthelloStateTest, SimpleFlippingTest)
{
    state->doMove(Move{4, 2});
    
    ASSERT_EQ(Field::White, state->playfield(4, 3));
    auto actualMoves = state->getPossibleMoves();
    
    // next the Black player's possible moves
    vector<Move> expectedMoves{
        Move{3, 2}, 
        Move{5, 2}, 
        Move{5, 4}
    };
    ASSERT_EQ(Player::Black, state->getCurrentPlayer());
    verifyMoves(expectedMoves, actualMoves);
}

TEST_F(OthelloStateTest, FlippingTest)
{
    state->doMove(Move{4, 2}); // White
    state->doMove(Move{5, 4}); // Black
    state->doMove(Move{3, 5}); // White
    state->doMove(Move{4, 1}); // Black

    ASSERT_EQ(Player::White, state->getCurrentPlayer());
    vector<Field> playfield = {
        F, F, F, F, F, F, F, F,
        F, F, F, F, B, F, F, F,
        F, F, F, F, B, F, F, F,
        F, F, F, W, B, F, F, F,
        F, F, F, W, B, B, F, F,
        F, F, F, W, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
    verifyPlayfield(playfield, *state);

    auto actualMoves = state->getPossibleMoves();
    vector<Move> expectedMoves = {
        Move{5, 1},
        Move{5, 2},
        Move{5, 3},
        Move{6, 4},
        Move{5, 5}
    };
    verifyMoves(expectedMoves, actualMoves);
}

TEST_F(OthelloStateTest, LockedStateTest)
{
    vector<Field> playfield = {
        B, F, W, W,
        F, B, B, B,
        W, B, B, B,
        W, B, B, W,
    };
    State lockedState(playfield, Player::White);

    ASSERT_EQ(Player::White, lockedState.getCurrentPlayer());
    auto possibleMoves = lockedState.getPossibleMoves();
    ASSERT_EQ(0U, possibleMoves.size());

    ASSERT_FALSE(lockedState.hasWon(Player::White));
    ASSERT_TRUE(lockedState.hasWon(Player::Black));
}

TEST_F(OthelloStateTest, LockedWhiteStateTest)
{
    vector<Field> playfield = {
        B, F, W, B,
        F, B, B, B,
        W, B, B, B,
        W, B, B, W,
    };
    State lockedWhiteState(playfield, Player::White);

    ASSERT_EQ(Player::White, lockedWhiteState.getCurrentPlayer());
    auto possibleMoves = lockedWhiteState.getPossibleMoves();
    ASSERT_EQ(0U, possibleMoves.size());

    State blackState(playfield, Player::Black);

    ASSERT_EQ(Player::Black, blackState.getCurrentPlayer());
    auto actualMoves = blackState.getPossibleMoves();
    verifyMoves({Move{1, 0}}, actualMoves);
}

TEST_F(OthelloStateTest, ChangeOfPossibleMovesTest)
{
    auto startMoves = state->getPossibleMoves();
    
    state->doMove(Move{4, 2}); // White
    state->doMove(Move{5, 4}); // Black
    state->doMove(Move{3, 5}); // White
    state->doMove(Move{4, 1}); // Black
    
    auto endMoves = state->getPossibleMoves();
    ASSERT_NE(startMoves.size(), endMoves.size());
}

TEST_F(OthelloStateTest, ChangeOfPlayersAfterMoveTest)
{
    state->doMove(Move{3, 5});
    
    ASSERT_EQ(Player::Black, state->getCurrentPlayer());

    vector<Field> expectedField = {
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, W, B, F, F, F,      
        F, F, F, W, W, F, F, F,      
        F, F, F, W, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
    verifyPlayfield(expectedField, *state);
    auto actualMoves = state->getPossibleMoves();
    vector<Move> expectedMoves = {
        Move{2, 3}, 
        Move{4, 5},
        Move{2, 5}
    };
    verifyMoves(expectedMoves, actualMoves);
