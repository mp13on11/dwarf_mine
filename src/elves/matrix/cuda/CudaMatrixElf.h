#pragma once

#include "matrix/MatrixElf.h"

class CudaMatrixElf : public MatrixElf
{
public:
	virtual void run(std::istream& input, std::ostream& output);
};
