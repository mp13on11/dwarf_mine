#pragma once

#include "ElfCategory.h"
#include "Scheduler.h"

#include <functional>
#include <memory>
#include <string>

class SchedulerFactory
{
public:
    static std::unique_ptr<SchedulerFactory> createFor(const std::string& type, const ElfCategory& category);

    ~SchedulerFactory();

    std::unique_ptr<Scheduler> createScheduler() const;

protected:
    static void validateType(const std::string& type);
    static void validateCategory(const ElfCategory& category);

    template<typename SchedulerType, typename ElfType>
    static std::function<Scheduler*()> createFactory();

    SchedulerFactory(const std::function<Scheduler*()>& factory);

private:
    static std::function<Scheduler*()> createFactory(const std::string& type, const ElfCategory& category);
    static std::function<Scheduler*()> createSmpFactory(const ElfCategory& category);
#ifdef HAVE_CUDA
    static std::function<Scheduler*()> createCudaFactory(const ElfCategory& category);
#endif

    std::function<Scheduler*()> factory;
};

template<typename SchedulerType, typename ElfType>
std::function<Scheduler*()> SchedulerFactory::createFactory()
{
	return []()
		{
			return new SchedulerType([]() { return new ElfType(); });
		};
}
