
#include "OthelloStateTest.h"
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

ostream& operator<<(ostream& out, const vector<OthelloMove>& moves)
{
    for ( const auto& move : moves)
    {
        out << "{"<<move.x<<", "<<move.y<<"}\n";
    }
    return out;
}

TEST_F(OthelloStateTest, SimpleFlippingTest)
{
    state->doMove(OthelloMove{4, 2});

    ASSERT_EQ(Field::White, state->atPosition(4, 3));
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

#define _F Field::Free
#define _W Field::White
#define _B Field::Black


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

TEST_F(OthelloStateTest, FlippingTest)
{
    state->doMove(OthelloMove{4, 2}); // White
    state->doMove(OthelloMove{5, 4}); // Black
    state->doMove(OthelloMove{3, 5}); // White
    state->doMove(OthelloMove{4, 1}); // Black

    ASSERT_EQ(Player::White, state->getCurrentPlayer());
    vector<Field> playfield = {
        _F, _F, _F, _F, _F, _F, _F, _F,
        _F, _F, _F, _F, _B, _F, _F, _F,
        _F, _F, _F, _F, _B, _F, _F, _F,
        _F, _F, _F, _W, _B, _F, _F, _F,
        _F, _F, _F, _W, _B, _B, _F, _F,
        _F, _F, _F, _W, _F, _F, _F, _F,
        _F, _F, _F, _F, _F, _F, _F, _F,
        _F, _F, _F, _F, _F, _F, _F, _F
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
        _B, _F, _W, _W,
        _F, _B, _B, _B,
        _W, _B, _B, _B,
        _W, _B, _B, _W,
    };
    OthelloState lockedState(playfield, Player::White);

    ASSERT_EQ(Player::White, lockedState.getCurrentPlayer());
    auto possibleMoves = lockedState.getPossibleMoves();
    ASSERT_FALSE(lockedState.hasPossibleMoves());
    ASSERT_EQ(0U, possibleMoves.size());

    ASSERT_FALSE(lockedState.hasWon(Player::White));
    ASSERT_TRUE(lockedState.hasWon(Player::Black));
}

TEST_F(OthelloStateTest, LockedWhiteStateTest)
{
    vector<Field> playfield = {
        _B, _F, _W, _B,
        _F, _B, _B, _B,
        _W, _B, _B, _B,
        _W, _B, _B, _W,
    };
    OthelloState lockedWhiteState(playfield, Player::White);

    ASSERT_EQ(Player::White, lockedWhiteState.getCurrentPlayer());
    auto possibleMoves = lockedWhiteState.getPossibleMoves();
    ASSERT_FALSE(lockedWhiteState.hasPossibleMoves());

    OthelloState blackState(playfield, Player::Black);

    ASSERT_EQ(Player::Black, blackState.getCurrentPlayer());
    auto actualMoves = blackState.getPossibleMoves();
    verifyMoves({OthelloMove{1, 0}}, actualMoves);
}