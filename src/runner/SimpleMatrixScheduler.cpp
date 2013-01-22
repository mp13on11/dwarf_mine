#include "SimpleMatrixScheduler.h"
#include "matrix/MatrixElf.h"

using namespace std;

SimpleMatrixScheduler::SimpleMatrixScheduler(const function<ElfPointer()>& factory) :
		MatrixScheduler(factory)
{
}

void SimpleMatrixScheduler::doDispatch()
{
	result = elf().multiply(left, right);
}
