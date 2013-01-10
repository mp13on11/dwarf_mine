#pragma once

struct OthelloMove
{
	int x;
	int y;

	OthelloMove& operator+=(const OthelloMove& other);
	OthelloMove operator+(const OthelloMove& second) const;
};

inline OthelloMove OthelloMove::operator+(const OthelloMove& second) const
{
	return OthelloMove{x + second.x, y + second.y};
}

inline OthelloMove& OthelloMove::operator+=(const OthelloMove& other)
{
	x += other.x;
	y += other.y;
	return *this;
}