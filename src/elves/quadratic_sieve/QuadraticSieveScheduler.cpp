#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"

using namespace std;

QuadraticSieveScheduler::QuadraticSieveScheduler(const std::function<ElfPointer()>& factory) :
    SchedulerTemplate(factory)
{

}

void QuadraticSieveScheduler::provideData(ProblemStatement& )
{

}

void QuadraticSieveScheduler::outputData(ProblemStatement& )
{

}

bool QuadraticSieveScheduler::hasData() const
{
    return false;
}

void QuadraticSieveScheduler::doSimpleDispatch()
{

}

void QuadraticSieveScheduler::doDispatch()
{

}
