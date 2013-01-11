#include "factorize/smp/SmpFactorizationElf.h"
#include "factorize/FactorizationScheduler.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixScheduler.h"
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
    else
        return unique_ptr<Elf>(new SmpFactorizationElf());
}

unique_ptr<Scheduler> SMPElfFactory::createSchedulerImplementation() const
{
    if (_category == "matrix")
        return unique_ptr<Scheduler>(new MatrixScheduler());
    else
        return unique_ptr<Scheduler>(new FactorizationScheduler());
}
