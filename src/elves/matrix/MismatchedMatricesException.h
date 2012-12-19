#pragma once

#include <stdexcept>
#include <string>

class MismatchedMatricesException : public std::runtime_error
{
public:
    MismatchedMatricesException(size_t leftColumns, size_t rightRows);

private:
    static std::string constructMessage(size_t leftColumns, size_t rightRows);
};
