#pragma once

#include <iostream>

struct OthelloMove
{
	int x;
	int y;

	OthelloMove& operator+=(const OthelloMove& other);
	OthelloMove operator+(const OthelloMove& second) const;

    friend bool operator==(const OthelloMove& first, const OthelloMove& second);

    friend std::ostream& operator<<(std::ostream& stream, const OthelloMove& move);
};

inline bool operator==(const OthelloMove& first, const OthelloMove& second)
{
    return first.x == second.x && first.y == second.y;
}

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