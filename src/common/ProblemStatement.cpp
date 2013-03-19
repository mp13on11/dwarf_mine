#include "ProblemStatement.h"
#include <iostream>

std::unique_ptr<std::iostream> ProblemStatement::streamOnFile(const std::string& filename, std::ios::openmode mode)
{
    std::unique_ptr<std::fstream> stream(new std::fstream(filename, mode));
    if (!stream->is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    return std::unique_ptr<std::iostream>(static_cast<std::iostream*>(stream.release()));
}
