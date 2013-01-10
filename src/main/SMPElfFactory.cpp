#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixScheduler.h"
#include "montecarlo/smp/SMPMonteCarloElf.h"
#include "main/SMPElfFactory.h"

using namespace std;

SMPElfFactory::SMPElfFactory(const ElfCategory& category) :
    ElfFactory(category)
{
}

unique_ptr<Elf> SMPElfFactory::createElfImplementation() const
{
    if (_category == "matrix")
        return unique_ptr<Elf>(new SMPMatrixElf());
    else if (_category == "montecarlo")
        return unique_ptr<Elf>(new SMPMonteCarloElf());
    else
        throw runtime_error("createElfImplementation(): category must be one of matrix|montecarlo");

}

unique_ptr<Scheduler> SMPElfFactory::createSchedulerImplementation() const
{
    return unique_ptr<Scheduler>(new MatrixScheduler());
}
