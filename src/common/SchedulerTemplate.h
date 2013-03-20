#pragma once

#include "BenchmarkResults.h"
#include "Communicator.h"
#include "Scheduler.h"

#include <functional>
#include <memory>
#include <stdexcept>

template<typename ElfType>
class SchedulerTemplate : public Scheduler
{
public:
    typedef ElfType *ElfPointer;

    SchedulerTemplate(const Communicator& communicator, const std::function<ElfPointer()>& factory);
    virtual ~SchedulerTemplate() = 0;

    virtual void dispatch();
    virtual void dispatchSimple();
    virtual void dispatchBenchmark(int node);

protected:
    Communicator communicator;
    BenchmarkResult nodeSet;

    virtual void doDispatch() = 0;
    virtual void doSimpleDispatch() = 0;
    virtual void doBenchmarkDispatch(int node) = 0;
    virtual bool hasData() const = 0;
    ElfType& elf() const;

private:
    std::unique_ptr<ElfType> _elf;
    std::function<ElfPointer()> _factory;

    void validate() const;
};

template<typename ElfType>
SchedulerTemplate<ElfType>::SchedulerTemplate(const Communicator& communicator, const std::function<ElfPointer()>& factory) :
    communicator(communicator), _factory(factory)
{
    for (size_t i=0; i<communicator.size(); ++i)
        nodeSet[i] = communicator.weights()[i];
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
void SchedulerTemplate<ElfType>::dispatchBenchmark(int node)
{
    _elf.reset(_factory());
    validate();
    doBenchmarkDispatch(node);
    _elf.release();
}

template<typename ElfType>
void SchedulerTemplate<ElfType>::validate() const
{
    if (communicator.isMaster())
    {
        if (!hasData())
        {
            throw std::runtime_error("SchedulerTemplate::dispatch(): No input data provided or generated!");
        }
    }
}
