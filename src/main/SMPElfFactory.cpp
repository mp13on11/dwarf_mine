#include "factorize/smp/SmpFactorizationElf.h"
#include "factorize/FactorizationScheduler.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixScheduler.h"
#include "montecarlo/MonteCarloScheduler.h"
#include "montecarlo/smp/SMPMonteCarloElf.h"
#include "main/SMPElfFactory.h"

using namespace std;

SMPElfFactory::SMPElfFactory(const ElfCategory& category) :
    ElfFactory(category)
{
}

unique_ptr<Scheduler> SMPElfFactory::createSchedulerImplementation() const
{
    if (_category == "matrix")
    {
        return unique_ptr<Scheduler>(
                new MatrixScheduler(
                        []() { return new SMPMatrixElf(); }
                    )
            );
    }
	else if (_category == "montecarlo")
	{
		return unique_ptr<Scheduler>(
				new MonteCarloScheduler(
						[]() { return new SMPMonteCarloElf(); }
					)
			);
	}
    else
    {
        return unique_ptr<Scheduler>(
                new FactorizationScheduler(
                        []() { return new SmpFactorizationElf(); }
                    )
            );
    }
}
