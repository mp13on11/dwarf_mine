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

#pragma once

#include "Move.h"
#include "OthelloUtil.h"
#include <vector>
#include <memory>
#include <set>
#include <iosfwd>
#include "Exceptions.h"

class State
{
public:
    State(size_t sideLength = 8);
    State(const State& state);
    State(const State& state, const Move& move);
    State(const Playfield& playfield, Player nextPlayer);

    void doMove(const Move& move);
    void passMove();
    MoveList getPossibleMoves() const;
    Player getCurrentPlayer() const;
    Player getCurrentEnemy() const;
    bool hasWon(Player player) const;

    int playfieldSideLength() const;
    const Field* playfieldBuffer() const;
    Field& playfield(size_t x, size_t y);
    Field playfield(size_t x, size_t y) const;
    bool onBoard(const Move& move) const;

private:
    std::vector<Field> _playfield;
    int _sideLength;
    Player _player;

    void flipCounters(const std::vector<Move>& counterPositions, Player player);
    MoveList getAllEnclosedCounters(const Move& move) const;
    MoveList getAdjacentEnemyDirections(const Move& move) const;
    MoveList getEnclosedCounters(const Move& move, const Move& direction) const;
    bool existsEnclosedCounters(const Move& move) const;
    Field& playfield(const Move& move);
    Field playfield(const Move& move) const;
    void ensureValidPlayfield() const;

    friend std::ostream& operator<<(std::ostream& stream, const State& state);
};

inline void State::ensureValidPlayfield() const 
{
    if (_playfield.size() != size_t(_sideLength * _sideLength))
        throw InvalidFieldSizeException(_playfield.size());
}

inline int State::playfieldSideLength() const
{
    return _sideLength;
}

inline Field& State::playfield(size_t x, size_t y)
{
    return _playfield[y * _sideLength + x];
}

inline Field State::playfield(size_t x, size_t y) const
{
    return _playfield[y * _sideLength + x];
}

inline Field& State::playfield(const Move& move)
{
    return playfield(move.x, move.y);
}

inline Field State::playfield(const Move& move) const
{
    return playfield(move.x, move.y);
}

inline const Field* State::playfieldBuffer() const
{
    return _playfield.data();
}

inline Player State::getCurrentEnemy() const
{
    return _player == Field::Black ? Field::White : Field::Black;
}

inline Player State::getCurrentPlayer() const
{
    return _player;
}

inline bool State::onBoard(const Move& move) const
{
    return (move.x < _sideLength && move.x >= 0 && move.y < _sideLength && move.y >= 0);
