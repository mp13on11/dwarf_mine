#pragma once

#include "matrix/MatrixScheduler.h"

#include <functional>

class SimpleMatrixScheduler : public MatrixScheduler
{
public:
	SimpleMatrixScheduler(const std::function<ElfPointer()>& factory);

protected:
	virtual void doDispatch();
};
