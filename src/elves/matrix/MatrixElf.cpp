#include "MatrixElf.h"
#include "Matrix.h"
#include "MatrixHelper.h"

void MatrixElf::run(std::istream& input, std::ostream& output)
{
    MatrixT left = MatrixHelper::readMatrixFrom(input);
    MatrixT right = MatrixHelper::readMatrixFrom(input);

    MatrixHelper::validateMultiplicationPossible(left, right);

    MatrixT result = multiply(left, right);
    MatrixHelper::writeMatrixTo(output, result);
}
