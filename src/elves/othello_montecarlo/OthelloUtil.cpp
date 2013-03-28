#include "OthelloUtil.h"
#include <Result.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace OthelloHelper
{
    size_t generateUniqueSeed(size_t nodeId, size_t threadId, size_t commonSeed)
    {
        std::stringstream buffer;
        buffer << commonSeed << " ~ "<< nodeId << " ~ " << threadId;
        std::hash<std::string> generateHash;
        return generateHash(buffer.str());
    }

    void writeResultToStream(std::ostream& stream, const Result& result)
    {
        stream << result.x 
               << result.y 
               << result.visits 
               << result.wins; 
    }

    void readResultFromStream(std::istream& stream, Result& result)
    {
        stream >> result.x
               >> result.y
               >> result.visits
               >> result.wins;
    }

    void writePlayfieldToStream(std::ostream& stream, const std::vector<Field>& playfield)
    {
        for (const auto& field : playfield)
        {
            stream << field;
        }
    }

    void readPlayfieldFromStream(std::istream& stream, std::vector<Field>& playfield)
    {
        if (stream.good())
        {
            Field field;
            stream >> field;
            while (stream.good())
            {
                playfield.push_back(field);
                stream >> field;
            }
        }
        else
        {
            throw std::runtime_error("Unexpected empty stream");
        }
        if (playfield.size() != 64)
        {
            throw std::runtime_error("Invalid playfield size");
        }
    }
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
    else if (field == Field::Free)
    {
        stream << "F";
    }
    else 
    {
        stream << "?";
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
        default:
            field = Field::Illegal;
    }
    return stream;
}
