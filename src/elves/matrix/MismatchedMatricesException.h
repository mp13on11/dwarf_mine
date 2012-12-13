#pragma once

#include <exception>
#include <string>

class MismatchedMatricesException : public std::exception
{
public:
    MismatchedMatricesException(size_t leftColumns, size_t rightRows);
    virtual ~MismatchedMatricesException() throw();
    virtual const char* what() const throw();

private:
    std::string message;

    void addToMessage(size_t number);
};
