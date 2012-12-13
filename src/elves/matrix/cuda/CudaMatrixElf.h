#pragma once

#include "elves/matrix/MatrixElf.h"

class CudaMatrixElf : public MatrixElf
{
public:
	virtual void run();
};
