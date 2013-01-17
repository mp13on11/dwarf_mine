#pragma once

#include <iostream>

struct OthelloMove
{
	int x;
	int y;

	OthelloMove& operator+=(const OthelloMove& other);
	OthelloMove operator+(const OthelloMove& second) const;

    friend std::ostream& operator<<(std::ostream& stream, const OthelloMove& move);
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

inline std::ostream& operator<<(std::ostream& stream, const OthelloMove& move)
{
    stream << "{" << move.x << ", " << move.y << "}";
    return stream;
}