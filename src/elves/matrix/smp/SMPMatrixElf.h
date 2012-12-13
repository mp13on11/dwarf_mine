#pragma once

#include <matrix/MatrixElf.h>
#include <functional>
#include <iostream>

class SMPMatrixElf : public MatrixElf
{

public:
	void run(std::istream& in, std::ostream& out);
};
