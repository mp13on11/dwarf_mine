#include <mpi.h>
#include <iostream>

using namespace std;

int main()
{
	MPI::Init();

	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();

	cout << "Hello from rank " << rank << " of " << size << "!" << endl;

	MPI::Finalize();
}
