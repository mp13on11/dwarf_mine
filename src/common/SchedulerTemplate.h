#pragma once

#include "Scheduler.h"

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
    ElfType& elf() const;

private:
    std::unique_ptr<ElfType> _elf;
    std::function<ElfPointer()> _factory;
};

template<typename ElfType>
SchedulerTemplate<ElfType>::SchedulerTemplate(const std::function<ElfPointer()>& factory) :
    _factory(factory)
{
}

template<typename ElfType>
SchedulerTemplate<ElfType>::~SchedulerTemplate()
{
}

template<typename ElfType>
ElfType& SchedulerTemplate<ElfType>::elf() const
{
    return *_elf;
}

template<typename ElfType>
void SchedulerTemplate<ElfType>::dispatch()
{
    _elf.reset(_factory());
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
    _elf.release();
}
