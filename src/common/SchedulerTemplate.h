#pragma once

#include "MpiHelper.h"
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
    virtual void dispatchSimple();

protected:
    virtual void doDispatch() = 0;
    virtual void doSimpleDispatch() = 0;
    virtual bool hasData() const = 0;
    ElfType& elf() const;

private:
    std::unique_ptr<ElfType> _elf;
    std::function<ElfPointer()> _factory;

    void validate() const;
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
#include <iostream>
template<typename ElfType>
void SchedulerTemplate<ElfType>::dispatch()
{
    _elf.reset(_factory());
    validate();
    doDispatch();
    _elf.release();
}

template<typename ElfType>
void SchedulerTemplate<ElfType>::dispatchSimple()
{
    _elf.reset(_factory());
    validate();
    doSimpleDispatch();
    _elf.release();
}

template<typename ElfType>
void SchedulerTemplate<ElfType>::validate() const
{
    if (MpiHelper::isMaster())
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
}
