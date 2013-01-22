#pragma once

#include "common/SchedulerFactory.h"

#include <functional>
#include <memory>

class SimpleSchedulerFactory : public SchedulerFactory
{
public:
	static std::unique_ptr<SchedulerFactory> createFor(const std::string& type, const ElfCategory& category);

	SimpleSchedulerFactory(const std::function<Scheduler*()>& factory);

private:
    static std::function<Scheduler*()> createFactory(const std::string& type, const ElfCategory& category);
    static std::function<Scheduler*()> createSmpFactory(const ElfCategory& category);
#ifdef HAVE_CUDA
    static std::function<Scheduler*()> createCudaFactory(const ElfCategory& category);
#endif
};
