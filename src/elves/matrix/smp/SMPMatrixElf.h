#pragma once

#include "matrix/MatrixElf.h"

class SMPMatrixElf : public MatrixElf
{

public:
	virtual void run(std::istream& in, std::ostream& out);
};
