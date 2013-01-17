#pragma once

#include "OthelloMove.h"
#include <vector>
#include <memory>
#include <set>
#include <iosfwd>


enum class Field { Free, Black, White };

typedef Field Player;

class OthelloState
{
public:
	OthelloState(const OthelloState& state);
	OthelloState(const std::vector<Field>& playfield, Player nextPlayer);
	OthelloState(size_t sideLength = 8);

	void doMove(const OthelloMove& move);
	std::vector<OthelloMove> getPossibleMoves();
	Player getCurrentPlayer() const;
	Player getCurrentEnemy() const;
	bool hasPossibleMoves();
	bool hasWon(Player player);

	Field atPosition(int x, int y);
	int playfieldSideLength() const;

private:
	// row - column
	std::vector<Field> _playfield;
	int _sideLength;
	Player _player;

	bool onBoard(const OthelloMove& move) const;
	std::vector<OthelloMove> getAllEnclosedCounters(const OthelloMove& move);
	std::vector<OthelloMove> getAdjacentEnemyDirections(const OthelloMove& move);
	std::vector<OthelloMove> getEnclosedCounters(const OthelloMove& move, const OthelloMove& direction);
	bool existsEnclosedCounters(const OthelloMove& move);
	Field& playfield(const OthelloMove& move);

	friend std::ostream& operator<<(std::ostream& stream, const OthelloState& state);
};

inline int OthelloState::playfieldSideLength() const
{
	return _sideLength;
}

inline Field& OthelloState::playfield(const OthelloMove& move)
{
	return _playfield[move.y * _sideLength + move.x];
}
