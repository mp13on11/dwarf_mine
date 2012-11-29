#include "benchmark/PerformanceEvent.h"
#include <algorithm>
#include <papi.h>

PerformanceEvent::PerformanceEvent(const std::string& name)
: name(name)
{
    generatedCode = false;
}

PerformanceEvent::PerformanceEvent(const std::string& name, const int code)
: PerformanceEvent(name)
{
    this->code = code;
    generatedCode = true;
}

PerformanceEvent::~PerformanceEvent()
{
}

const std::string& PerformanceEvent::getName() const
{
    return name;
}

int PerformanceEvent::getCode()
{
    if (!generatedCode)
        generateCode();
    return code;
}

void PerformanceEvent::generateCode()
{
    char* convertedName = getNameAsCharArray();
    PAPI_event_name_to_code(convertedName, &code);
    generatedCode = true;
    free(convertedName);
}

char* PerformanceEvent::getNameAsCharArray() const
{
    char* convertedName = new char[name.size() + 1];
    std::copy(name.cbegin(), name.cend(), convertedName);
    convertedName[name.size()] = '\0';
    return convertedName;
}
