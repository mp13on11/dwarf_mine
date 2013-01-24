#include "Configuration.h"
#include "Scheduler.h"
#include "SchedulerFactory.h"

using namespace std;

unique_ptr<Scheduler> Configuration::createScheduler() const
{
    return createSchedulerFactory()->createScheduler();
}
