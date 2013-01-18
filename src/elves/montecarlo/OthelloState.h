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
	OthelloState(const OthelloState& state);
	OthelloState(const std::vector<Field>& playfield, Player nextPlayer);
	OthelloState(size_t sideLength = 8);

	void doMove(const OthelloMove& move);
	std::vector<OthelloMove> getPossibleMoves();
	OthelloMove getRandomMove(RandomGenerator generator);
	Player getCurrentPlayer() const;
	Player getCurrentEnemy() const;
	bool hasPossibleMoves();
	bool hasWon(Player player);

	Field atPosition(int x, int y);
	int playfieldSideLength() const;
	Field* playfieldBuffer();

private:
	std::vector<Field> _playfield;
	int _sideLength;
	Player _player;

	void flipCounters(const std::vector<OthelloMove>& counterPositions, Player player);
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

inline Field* OthelloState::playfieldBuffer()
{
	return _playfield.data();
}