#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 256
#endif

using namespace std;

int main()
{
	MPI::Init();

	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();

	char host[HOST_NAME_MAX];
	int status = gethostname(host, HOST_NAME_MAX);
	string hostname;

	if (status == -1)
		hostname = "<unknown>";
	else
		hostname = host;

	cout << "Hello from rank " << rank << " of " << size << "! (I'm on host " << hostname << ".)" << endl;

	MPI::Finalize();
}
