#include "SimpleConfiguration.h"
#include "SimpleSchedulerFactory.h"

using namespace std;

SimpleConfiguration::SimpleConfiguration(int argc, char** argv) :
	CommandLineConfiguration(argc, argv)
{
}

unique_ptr<SchedulerFactory> SimpleConfiguration::createSchedulerFactory() const
{
	return SimpleSchedulerFactory::createFor(mode(), category());
}
