#pragma once

#include <iosfwd>
#include <vector>

struct OthelloResult
{
    size_t x;
    size_t y;
    size_t visits;
    size_t wins;

    bool equalMove(const OthelloResult& other) const
    {
        return (x == other.x && y == other.y);
    }

    double successRate() const
    {
        return 1.0 * wins / visits;
    }
};

enum class Field { Free, Black, White };

// shortcuts
#define F Field::Free
#define W Field::White
#define B Field::Black

typedef Field Player;

std::ostream& operator<<(std::ostream& stream, OthelloResult& result);

std::istream& operator>>(std::istream& stream, OthelloResult& result);

std::ostream& operator<<(std::ostream& stream, const std::vector<Field>& playfield);

std::istream& operator>>(std::istream& stream, std::vector<Field>& playfield);

std::ostream& operator<<(std::ostream& stream, const Field field);

std::istream& operator>>(std::istream& stream, Field& field);