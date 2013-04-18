/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#pragma once

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

protected:
    Communicator communicator;

    virtual void doDispatch() = 0;
    virtual void doSimpleDispatch() = 0;
    virtual bool hasData() const = 0;
    ElfType& elf() const;

private:
    std::unique_ptr<ElfType> _elf;
    std::function<ElfPointer()> _factory;

    void validate() const;
    void performDispatch();
};

template<typename ElfType>
SchedulerTemplate<ElfType>::SchedulerTemplate(const Communicator& communicator, const std::function<ElfPointer()>& factory) :
    communicator(communicator), _factory(factory)
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
    validate();
    performDispatch();
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

template<typename ElfType>
void SchedulerTemplate<ElfType>::performDispatch()
{
    if (communicator.isWorld() && communicator.size() == 1)
    {
        doSimpleDispatch();
    }
    else
    {
        doDispatch();
    }
}
