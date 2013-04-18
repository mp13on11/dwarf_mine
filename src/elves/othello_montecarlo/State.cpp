#include "State.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace std;

vector<Move> ADJACENT_DIRECTIONS = {
    {-1, 1}, { 0, 1}, { 1, 1},
    {-1, 0},          { 1, 0},
    {-1,-1}, { 0,-1}, { 1,-1}
};

State::State(const State& state)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{
    ensureValidPlayfield();
}

State::State(const State& state, const Move& move)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{
    ensureValidPlayfield();
    doMove(move);
}

State::State(const vector<Field>& playfield, Player nextPlayer)
    : _playfield(playfield), _sideLength(sqrt(playfield.size())), _player(nextPlayer)
{
    ensureValidPlayfield();
}

State::State(size_t sideLength)
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

void State::doMove(const Move& move)
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

void State::passMove()
{
    _player = getCurrentEnemy();
}

void State::flipCounters(const vector<Move>& counterPostions, Player player) 
{
    for (const auto& position : counterPostions)
    {
        playfield(position) = player;
    }
}

vector<Move> State::getAdjacentEnemyDirections(const Move& move) const
{
    vector<Move> directions;
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

vector<Move> State::getAllEnclosedCounters(const Move& move) const
{
    vector<Move> counters;
    /*_ADJACENT_DIRECTIONS(enemyDirections, move);*/
    vector<Move> enemyDirections = getAdjacentEnemyDirections(move);
    for (const auto& direction : enemyDirections)
    {
        vector<Move> enclosed = getEnclosedCounters(move, direction);
        for (const auto& e : enclosed)
            counters.push_back(e);
    }
    return counters;
}

MoveList State::getEnclosedCounters(const Move& move, const Move& direction) const
{
    auto position = move + direction;
    vector<Move> counters;
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

MoveList State::getPossibleMoves() const
{
    vector<Move> moves;

    for (size_t i = 0; i < _playfield.size(); ++i)
    {
        if (_playfield[i] == Field::Free)
        {
            Move position{int(i % _sideLength), int(i / _sideLength)};
            if (existsEnclosedCounters(position))
            {
                moves.push_back(move(position));
            }
        }
    }
    return moves;
}

bool State::existsEnclosedCounters(const Move& move) const
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

bool State::hasWon(Player player) const
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

ostream& operator<<(ostream& stream, const State& state)
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