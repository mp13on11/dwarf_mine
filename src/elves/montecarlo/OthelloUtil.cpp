#include "OthelloUtil.h"

#include <iostream>
#include <stdexcept>

std::ostream& operator<<(std::ostream& stream, OthelloResult& result)
{
    stream << result.x << result.y << result.visits << result.wins;
    return stream;
}

std::istream& operator>>(std::istream& stream, OthelloResult& result)
{
    stream >> result.x;
    stream >> result.y;
    stream >> result.visits;
    stream >> result.wins;
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<Field>& playfield)
{
    for (const auto& field : playfield)
    {
        stream << field;
    }
    return stream;
}

std::istream& operator>>(std::istream& stream, std::vector<Field>& playfield)
{
    if (stream.good())
    {
        do 
        {
            Field field;
            stream >> field;
            playfield.push_back(field);
        }
        while (stream.good());
    }
    else
    {
        throw std::runtime_error("Unexpected empty stream");
    }
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Field field)
{
    if (field == Field::Black)
    {
        stream << "B";
    }
    else if (field == Field::White)
    {
        stream << "W";
    }
    else 
    {
        stream << "F";
    }
    return stream;
}

std::istream& operator>>(std::istream& stream, Field& field)
{
    char sign;
    stream >> sign;
    switch (sign)
    {
        case 'B': field = Field::Black;
            break;
        case 'W': field = Field::White;
            break;
        case 'F': field = Field::Free;
            break;
    }
    return stream;
}