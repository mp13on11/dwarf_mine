
#include "OthelloStateTest.h"
#include "OthelloUtil.h"
#include <montecarlo/OthelloMove.h>
#include <montecarlo/OthelloExceptions.h>
#include <iostream>
#include <cmath>

using namespace std;

void OthelloStateTest::SetUp()
{
    state = new OthelloState(8);
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
    ASSERT_THROW(OthelloState(9), InvalidFieldSizeException);
}

TEST_F(OthelloStateTest, DoMoveThrowOnOutOfBoundsTest)
{
    ASSERT_THROW(state->doMove(OthelloMove{8, 8}), InvalidMoveException);
    ASSERT_THROW(state->doMove(OthelloMove{-1, -1}), InvalidMoveException);
    ASSERT_THROW(state->doMove(OthelloMove{5, 8}), InvalidMoveException);
    ASSERT_THROW(state->doMove(OthelloMove{3, -5}), InvalidMoveException);   
}

TEST_F(OthelloStateTest, DoMoveThrowOnAccessOccupiedFieldTest)
{
    ASSERT_THROW(state->doMove(OthelloMove{3, 3}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(OthelloMove{3, 4}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(OthelloMove{4, 3}), OccupiedFieldException);
    ASSERT_THROW(state->doMove(OthelloMove{4, 4}), OccupiedFieldException);
}


TEST_F(OthelloStateTest, GetPossibleMovesTest)
{
    auto actualMoves = state->getPossibleMoves();
    vector<OthelloMove> expectedMoves{
        OthelloMove{4, 2}, 
        OthelloMove{5, 3}, 
        OthelloMove{2, 4},
        OthelloMove{3, 5}
    };
    verifyMoves(expectedMoves, actualMoves);
}



TEST_F(OthelloStateTest, SimpleFlippingTest)
{
    state->doMove(OthelloMove{4, 2});
    
    ASSERT_EQ(Field::White, state->playfield(4, 3));
    auto actualMoves = state->getPossibleMoves();
    
    // next the Black player's possible moves
    vector<OthelloMove> expectedMoves{
        OthelloMove{3, 2}, 
        OthelloMove{5, 2}, 
        OthelloMove{5, 4}
    };
    ASSERT_EQ(Player::Black, state->getCurrentPlayer());
    verifyMoves(expectedMoves, actualMoves);
}

TEST_F(OthelloStateTest, FlippingTest)
{
    state->doMove(OthelloMove{4, 2}); // White
    state->doMove(OthelloMove{5, 4}); // Black
    state->doMove(OthelloMove{3, 5}); // White
    state->doMove(OthelloMove{4, 1}); // Black

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
    vector<OthelloMove> expectedMoves = {
        OthelloMove{5, 1},
        OthelloMove{5, 2},
        OthelloMove{5, 3},
        OthelloMove{6, 4},
        OthelloMove{5, 5}
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
    OthelloState lockedState(playfield, Player::White);

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
    OthelloState lockedWhiteState(playfield, Player::White);

    ASSERT_EQ(Player::White, lockedWhiteState.getCurrentPlayer());
    auto possibleMoves = lockedWhiteState.getPossibleMoves();
    ASSERT_EQ(0U, possibleMoves.size());

    OthelloState blackState(playfield, Player::Black);

    ASSERT_EQ(Player::Black, blackState.getCurrentPlayer());
    auto actualMoves = blackState.getPossibleMoves();
    verifyMoves({OthelloMove{1, 0}}, actualMoves);
}

TEST_F(OthelloStateTest, ChangeOfPossibleMovesTest)
{
    auto startMoves = state->getPossibleMoves();
    
    state->doMove(OthelloMove{4, 2}); // White
    state->doMove(OthelloMove{5, 4}); // Black
    state->doMove(OthelloMove{3, 5}); // White
    state->doMove(OthelloMove{4, 1}); // Black
    
    auto endMoves = state->getPossibleMoves();
    ASSERT_NE(startMoves.size(), endMoves.size());
}

TEST_F(OthelloStateTest, ChangeOfPlayersAfterMoveTest)
{
    state->doMove(OthelloMove{3, 5});
    
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
    vector<OthelloMove> expectedMoves = {
        OthelloMove{2, 3}, 
        OthelloMove{4, 5},
        OthelloMove{2, 5}
    };
    verifyMoves(expectedMoves, actualMoves);
}