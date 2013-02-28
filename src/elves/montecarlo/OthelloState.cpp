#include "OthelloState.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace std;

#define TEST_AND_ADD(VECTOR, MOVE, DIRECTION_X, DIRECTION_Y) \
    position = {MOVE.x + DIRECTION_X, MOVE.y + DIRECTION_Y};                              \
    if (onBoard(position) && playfield(position) == getCurrentEnemy())  \
    {                                                                   \
        VECTOR.push_back(OthelloMove{DIRECTION_X, DIRECTION_Y});        \
    }

#define ADJACENT_DIRECTIONS(VECTOR, MOVE)   \
    vector<OthelloMove> VECTOR;             \
    OthelloMove position;                    \
    TEST_AND_ADD(VECTOR, MOVE, -1, 1)       \
    TEST_AND_ADD(VECTOR, MOVE, -1, 0)       \
    TEST_AND_ADD(VECTOR, MOVE, -1,-1)       \
    TEST_AND_ADD(VECTOR, MOVE,  0, 1)       \
    TEST_AND_ADD(VECTOR, MOVE,  0,-1)       \
    TEST_AND_ADD(VECTOR, MOVE,  1, 1)       \
    TEST_AND_ADD(VECTOR, MOVE,  1, 0)       \
    TEST_AND_ADD(VECTOR, MOVE,  1,-1)

vector<OthelloMove> ADJACENT_DIRECTIONS = {
    {-1, 1}, { 0, 1}, { 1, 1},
    {-1, 0},          { 1, 0},
    {-1,-1}, { 0,-1}, { 1,-1}
};

OthelloState::OthelloState(const OthelloState& state)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{
    ensureValidPlayfield();
}

OthelloState::OthelloState(const OthelloState& state, const OthelloMove& move)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{
    ensureValidPlayfield();
    doMove(move);
}

OthelloState::OthelloState(const vector<Field>& playfield, Player nextPlayer)
    : _playfield(playfield), _sideLength(sqrt(playfield.size())), _player(nextPlayer)
{
    ensureValidPlayfield();
}

OthelloState::OthelloState(size_t sideLength)
    : _sideLength((int)sideLength), _player(Player::White)
{
    if (_sideLength % 2 != 0)
    {
        throw InvalidFieldSizeException(sideLength);
    }

    _playfield.resize(_sideLength * _sideLength);
    int center = _sideLength / 2 - 1;
    playfield(center    , center    ) = Field::White; // top left
    playfield(center    , center + 1) = Field::Black; // top right
    playfield(center + 1, center    ) = Field::Black; // bottom left
    playfield(center + 1, center + 1) = Field::White; // bottom right

    ensureValidPlayfield();
}

void OthelloState::doMove(const OthelloMove& move)
{
    if (!onBoard(move))
    {
        throw InvalidMoveException((size_t)_sideLength, move);
    }
    if (playfield(move) != Field::Free)
    {
        throw OccupiedFieldException(move);
    }

    auto enclosedCounterPositions = getAllEnclosedCounters(move);

    playfield(move) = _player;
    flipCounters(enclosedCounterPositions, _player);

    _player = getCurrentEnemy();
}

void OthelloState::passMove()
{
    _player = getCurrentEnemy();
}

void OthelloState::flipCounters(const vector<OthelloMove>& counterPostions, Player player) 
{
    for (const auto& position : counterPostions)
    {
        playfield(position) = player;
    }
}

vector<OthelloMove> OthelloState::getAdjacentEnemyDirections(const OthelloMove& move) const
{
    vector<OthelloMove> directions;
    for (const auto& direction : ADJACENT_DIRECTIONS)
    {
        const auto position = move + direction;
        if (onBoard(position) && playfield(position) == getCurrentEnemy())
        {
            directions.push_back(direction);
        }
    }
    return directions;
}

vector<OthelloMove> OthelloState::getAllEnclosedCounters(const OthelloMove& move) const
{
    vector<OthelloMove> counters;
    /*_ADJACENT_DIRECTIONS(enemyDirections, move);*/
    vector<OthelloMove> enemyDirections = getAdjacentEnemyDirections(move);
    for (const auto& direction : enemyDirections)
    {
        vector<OthelloMove> enclosed = getEnclosedCounters(move, direction);
        for (const auto& e : enclosed)
            counters.push_back(e);
    }
    return counters;
}

MoveList OthelloState::getEnclosedCounters(const OthelloMove& move, const OthelloMove& direction) const
{
    auto position = move + direction;
    vector<OthelloMove> counters;
    while (onBoard(position) && playfield(position) == getCurrentEnemy())
    {
        counters.push_back(position);
        position += direction;
    }
    if (onBoard(position) && playfield(position) == getCurrentPlayer())
    {
        return counters;
    }
    counters.clear();
    return counters;
}

MoveList OthelloState::getPossibleMoves() const
{
    vector<OthelloMove> moves;

    for (size_t i = 0; i < _playfield.size(); ++i)
    {
        if (_playfield[i] == Field::Free)
        {
            OthelloMove position{int(i % _sideLength), int(i / _sideLength)};
            if (existsEnclosedCounters(position))
            {
                moves.push_back(move(position));
            }
        }
    }
    return moves;
}

bool OthelloState::existsEnclosedCounters(const OthelloMove& move) const
{
    auto directions = getAdjacentEnemyDirections(move);
    for (const auto& direction : directions)
    {
        if (getEnclosedCounters(move, direction).size() > 0)
        {
            return true;
        }
    }
    return false;
}

bool OthelloState::hasWon(Player player) const
{
    int superiority = 0;
    for (size_t i = 0; i < _playfield.size(); ++i)
    {
        if (_playfield[i] == player) 
        {
            ++superiority;
        }
        else if (_playfield[i] != Field::Free)
        { 
            --superiority;
        }
    }
    return superiority >= 0;
}

ostream& operator<<(ostream& stream, const OthelloState& state)
{
    stream << "  ";
    for (int i = 0; i < state._sideLength; ++i)
    {
        stream << " " << i;
    }
    stream << "\n";
    for (int i = 0; i < state._sideLength; ++i)
    {
        stream << " "<< i;
        for (int j = 0; j < state._sideLength; ++j) 
        {
            auto field = state._playfield[i * state._sideLength + j];
            if (field != Field::Free)
                stream << " " << field;
            else
                stream << "  ";
        }
        stream << "\n";
    }
    return stream;
}