#pragma once

#include "main/Scheduler.h"

#include <functional>
#include <memory>
#include <stdexcept>

template<typename ElfType>
class SchedulerTemplate : public Scheduler
{
public:
	typedef ElfType *ElfPointer;

	SchedulerTemplate(const std::function<ElfPointer()>& factory);
    virtual ~SchedulerTemplate() = 0;

    virtual void dispatch();

protected:
    virtual void doDispatch() = 0;
    virtual bool hasData() = 0;

    std::unique_ptr<ElfType> elf;
};

template<typename ElfType>
SchedulerTemplate<ElfType>::SchedulerTemplate(const std::function<ElfPointer()>& factory) :
	elf(factory())
{
}

template<typename ElfType>
SchedulerTemplate<ElfType>::~SchedulerTemplate()
{
}

template<typename ElfType>
void SchedulerTemplate<ElfType>::dispatch()
{
    if (MpiHelper::isMaster(rank))
    {
        if (!hasData())
        {
            throw std::runtime_error("SchedulerTemplate::dispatch(): No ProblemStatement configured!");
        }

        if (nodeSet.empty())
        {
            throw std::runtime_error("SchedulerTemplate::dispatch(): Nodeset is empty!");
        }
    }

    doDispatch();
}
