#pragma once

#include "OthelloMove.h"
#include "OthelloUtil.h"
#include <vector>
#include <memory>
#include <set>
#include <iosfwd>
#include "OthelloExceptions.h"

class OthelloState
{
public:
    OthelloState(size_t sideLength = 8);
    OthelloState(const OthelloState& state);
    OthelloState(const OthelloState& state, const OthelloMove& move);
    OthelloState(const Playfield& playfield, Player nextPlayer);

    void doMove(const OthelloMove& move);
    void passMove();
    MoveList getPossibleMoves() const;
    Player getCurrentPlayer() const;
    Player getCurrentEnemy() const;
    bool hasWon(Player player) const;

    int playfieldSideLength() const;
    const Field* playfieldBuffer() const;
    Field& playfield(size_t x, size_t y);
    Field playfield(size_t x, size_t y) const;
    bool onBoard(const OthelloMove& move) const;

private:
    std::vector<Field> _playfield;
    int _sideLength;
    Player _player;

    void flipCounters(const std::vector<OthelloMove>& counterPositions, Player player);
    MoveList getAllEnclosedCounters(const OthelloMove& move) const;
    MoveList getAdjacentEnemyDirections(const OthelloMove& move) const;
    MoveList getEnclosedCounters(const OthelloMove& move, const OthelloMove& direction) const;
    bool existsEnclosedCounters(const OthelloMove& move) const;
    Field& playfield(const OthelloMove& move);
    Field playfield(const OthelloMove& move) const;
    void ensureValidPlayfield() const;

    friend std::ostream& operator<<(std::ostream& stream, const OthelloState& state);
};

inline void OthelloState::ensureValidPlayfield() const 
{
    if (_playfield.size() != size_t(_sideLength * _sideLength))
        throw InvalidFieldSizeException(_playfield.size());
}

inline int OthelloState::playfieldSideLength() const
{
    return _sideLength;
}

inline Field& OthelloState::playfield(size_t x, size_t y)
{
    return _playfield[y * _sideLength + x];
}

inline Field OthelloState::playfield(size_t x, size_t y) const
{
    return _playfield[y * _sideLength + x];
}

inline Field& OthelloState::playfield(const OthelloMove& move)
{
    return playfield(move.x, move.y);
}

inline Field OthelloState::playfield(const OthelloMove& move) const
{
    return playfield(move.x, move.y);
}

inline const Field* OthelloState::playfieldBuffer() const
{
    return _playfield.data();
}

inline Player OthelloState::getCurrentEnemy() const
{
    return _player == Field::Black ? Field::White : Field::Black;
}

inline Player OthelloState::getCurrentPlayer() const
{
    return _player;
}

inline bool OthelloState::onBoard(const OthelloMove& move) const
{
    return (move.x < _sideLength && move.x >= 0 && move.y < _sideLength && move.y >= 0);
}