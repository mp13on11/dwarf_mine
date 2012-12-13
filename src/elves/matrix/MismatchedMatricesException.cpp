#include "MismatchedMatricesException.h"

#include <sstream>

using namespace std;

MismatchedMatricesException::MismatchedMatricesException(size_t leftColumns, size_t rightRows) :
        message("Columns of left matrix must be equal to rows of right matrix. Left: ")
{
    addToMessage(leftColumns);
    message += "; right: ";
    addToMessage(rightRows);
}

MismatchedMatricesException::~MismatchedMatricesException() throw()
{
}

const char* MismatchedMatricesException::what() const throw()
{
    return message.c_str();
}

void MismatchedMatricesException::addToMessage(size_t number)
{
    stringstream stream;
    stream << number;
    message += stream.str();
}
