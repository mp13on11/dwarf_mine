#include "benchmark/PerformanceEvent.h"
#include "tools/PapiHelper.h"

#include <algorithm>

PerformanceEvent::PerformanceEvent(const std::string& name) :
    name(name), code(0), generatedCode(false)
{
}

PerformanceEvent::PerformanceEvent(const std::string& name, int code) :
    name(name), code(code), generatedCode(true)
{
}

PerformanceEvent::~PerformanceEvent()
{
}

const std::string& PerformanceEvent::getName() const
{
    return name;
}

int PerformanceEvent::getCode() const
{
    if (!generatedCode)
        generateCode();

    return code;
}

void PerformanceEvent::generateCode() const
{
    code = PapiHelper::eventCodeFrom(name);
    generatedCode = true;
}
