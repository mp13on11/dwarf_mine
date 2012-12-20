#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixScheduler.h"
#include "main/SMPElfFactory.h"

using namespace std;

SMPElfFactory::SMPElfFactory(const ElfCategory& category)
    : ElfFactory(category)
{

}

unique_ptr<Elf> SMPElfFactory::createElfImplementation() const
{
    return unique_ptr<Elf>(new SMPMatrixElf());
}

unique_ptr<Scheduler> SMPElfFactory::createSchedulerImplementation() const
{
    return unique_ptr<Scheduler>(new MatrixScheduler());
}
