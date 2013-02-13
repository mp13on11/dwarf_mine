#pragma once

#include "OthelloMove.h"
#include "OthelloUtil.h"
#include <vector>
#include <memory>
#include <set>
#include <iosfwd>

class OthelloState
{
public:
	OthelloState(size_t sideLength = 8);
	OthelloState(const OthelloState& state);
	OthelloState(const Playfield& playfield, Player nextPlayer);

	void doMove(const OthelloMove& move);
	MoveList getPossibleMoves() const;
	OthelloMove getRandomMove(RandomGenerator generator) const;
	Player getCurrentPlayer() const;
	Player getCurrentEnemy() const;
	bool hasPossibleMoves() const;
	bool hasWon(Player player) const;

	Field atPosition(int x, int y) const;
	int playfieldSideLength() const;
	Field* playfieldBuffer();
	Field playfield(size_t x, size_t y) const;

private:
	std::vector<Field> _playfield;
	int _sideLength;
	Player _player;

	void flipCounters(const std::vector<OthelloMove>& counterPositions, Player player);
	bool onBoard(const OthelloMove& move) const;
	MoveList getAllEnclosedCounters(const OthelloMove& move) const;
	MoveList getAdjacentEnemyDirections(const OthelloMove& move) const;
	MoveList getEnclosedCounters(const OthelloMove& move, const OthelloMove& direction) const;
	bool existsEnclosedCounters(const OthelloMove& move) const;
	Field& playfield(const OthelloMove& move);
	Field playfield(const OthelloMove& move) const;

	friend std::ostream& operator<<(std::ostream& stream, const OthelloState& state);
};

inline int OthelloState::playfieldSideLength() const
{
	return _sideLength;
}

inline Field OthelloState::playfield(size_t x, size_t y) const
{
	return _playfield[y * _sideLength + x];
}

inline Field& OthelloState::playfield(const OthelloMove& move)
{
	return _playfield[move.y * _sideLength + move.x];
}

inline Field OthelloState::playfield(const OthelloMove& move) const
{
	return _playfield[move.y * _sideLength + move.x];
}

inline Field* OthelloState::playfieldBuffer()
{
	return _playfield.data();
}