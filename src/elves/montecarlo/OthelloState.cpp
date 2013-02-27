#include "OthelloState.h"
#include "OthelloExceptions.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace std;

vector<OthelloMove> DIRECTIONS = {
    {-1, 1}, { 0, 1}, { 1, 1},
    {-1, 0},          { 1, 0},
    {-1,-1}, { 0,-1}, { 1,-1}
};

OthelloState::OthelloState(const OthelloState& state)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{

}

OthelloState::OthelloState(const OthelloState& state, const OthelloMove& move)
    : _playfield(state._playfield), _sideLength(state._sideLength), _player(state._player)
{
    doMove(move);
}

OthelloState::OthelloState(const vector<Field>& playfield, Player nextPlayer)
    : _playfield(playfield), _sideLength(sqrt(playfield.size())), _player(nextPlayer)
{
    if (_sideLength % 2 != 0)
    {
        throw InvalidFieldSizeException(_sideLength);
    }
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
    playfield({center    , center    }) = Field::White; // top left
    playfield({center    , center + 1}) = Field::Black; // top right
    playfield({center + 1, center    }) = Field::Black;
    playfield({center + 1, center + 1}) = Field::White;
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

bool OthelloState::onBoard(const OthelloMove& move) const
{
    return move.x < _sideLength && move.x >= 0 && move.y < _sideLength && move.y >= 0;
}

vector<OthelloMove> OthelloState::getAdjacentEnemyDirections(const OthelloMove& move) const
{
    vector<OthelloMove> directions;
    for (const auto& direction : DIRECTIONS)
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

OthelloMove OthelloState::getRandomMove(RandomGenerator generator) const
{
    auto moves = getPossibleMoves();
    return moves[generator(moves.size())];
}

MoveList OthelloState::getPossibleMoves() const
{
    vector<OthelloMove> moves;
    for (int i = 0; i < _sideLength; ++i)
    {
        for (int j = 0; j < _sideLength; ++j)
        {
            auto position = OthelloMove{i, j};
            if (playfield(position) == Field::Free
                && existsEnclosedCounters(position))
            {
                moves.push_back(move(position));
            }
        }
    }
    return moves;
}
// enclosed

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

// TODO optimize - cache of some fancy rule to detect a move
bool OthelloState::hasPossibleMoves() const
{
    return getPossibleMoves().size() > 0;
}

bool OthelloState::hasWon(Player player) const
{
    size_t playerCounter = 0;
    size_t enemyCounter = 0;
    for (int i = 0; i < _sideLength; ++i)
    {
        for (int j = 0; j < _sideLength; ++j)
        {
            OthelloMove position{i, j};
            Field field = playfield(position);
            if (field == Field::Free) 
                continue;
            if (field == player)
                playerCounter++;
            else 
                enemyCounter++;
        }
    }
    return playerCounter > enemyCounter;
}

Player OthelloState::getCurrentEnemy() const
{
    return _player == Field::Black ? Field::White : Field::Black;
}

Player OthelloState::getCurrentPlayer() const
{
    return _player;
}

Field OthelloState::atPosition(int x, int y) const
{
    return playfield(OthelloMove{x, y});
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