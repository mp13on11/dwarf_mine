#include "MismatchedMatricesException.h"

#include <sstream>

using namespace std;

MismatchedMatricesException::MismatchedMatricesException(size_t leftColumns, size_t rightRows) :
    runtime_error(constructMessage(leftColumns, rightRows))
{
}

string MismatchedMatricesException::constructMessage(size_t leftColumns, size_t rightRows)
{
    stringstream stream;
    stream
        << "Columns of left matrix must be equal to rows of right matrix. Left: "
        << leftColumns << "; right: " << rightRows;
    return stream.str();
}
