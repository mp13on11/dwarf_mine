Dwarf Mine - The 13-11 Benchmark
================================

This was a Master's project by the [Operating Systems and Middleware Group](http://www.dcl.hpi.uni-potsdam.de) 
at the [Hasso-Plattner-Institute (HPI)](http://www.hpi-web.de) in Potsdam, Germany.  
Our goal was to define and implement a **parallel benchmark suite for heterogeneous many-core systems**, 
based on the 13 [Berkeley Dwarfs](http://www.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.html).

Three representative parallel algorithms were chosen and implemented in a GPU and a CPU
variant:

* Matrix multiplication
* Quadratic Sieve
* Othello game simulation

Furthermore, a problem-specific scheduler was implemented for each of these algorithms, which uses MPI to distribute an
input problem among several physical machines, and to distribute the workload among all CPUs and GPUs in each machine.
So ultimately, every processor and graphics processor in a cluster are combined to solve a single problem. 
The run time of the whole calculation is measured and printed out at the end, possibly using the average of several runs.

Before the actual benchmark stage, each participating processor (GPU or CPU) is rated according to it's speed regarding
the chosen algorithm. 
These ratings are used to distribute the workload accordingly - faster processors get bigger slices of the input data 
than slower processors, thus maximizing processor utilization in a heterogeneous environment.

## Build instructions

This is a CMake based project, but it has only been compiled and tested in Linux environments so far. 
The CMake scripts assume GCC, but in principle, any compiler with sufficient C++ 11 support should work.
In order to build and run the GPU implementations, a Fermi-class NVIDIA graphics card is required.
Setting the `BUILD_WITH_CUDA` CMake option to `OFF` will disable building the CUDA implementations.

### Prerequisites/Dependencies

* GCC 4.7 or higher
* CUDA 5.0 or higher (optional)
* OpenMPI 1.6.4 or higher, _with multithreading support_
* Boost 1.49 or higher
* GMP 5.0.5 or higher, GMPXX C++ bindings

**NOTE:** The OpenMPI package provided in Debian's package management system is built without multithreading support and therefore is not compatible with this project. To run all scenarios successfully,
you have to build MPI from source:

```
wget http://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.4.tar.bz2
tar -xvf openmpi-1.6.4.tar.bz2
cd openmpi-1.6.4
./configure --prefix=/usr/lib/openmpi --enable-mpi-thread-multiple
make -j
sudo make install
sudo cp --symbolic-link /usr/lib/openmpi/bin/* /usr/bin/
sudo cp --symbolic-link /usr/lib/openmpi/lib/* /usr/lib/
```

### Building

Either run `build_all.sh`, which will generate a Debug build by default, or follow these steps:

```
mkdir build
cd build
cmake ..
make
```

After building, you should run all the tests using `test.sh` or `<BUILD_DIR>/src/test/test` if your
build directory is something else than `build`.

## Running/Example scenarios

A recorded shell session demonstrating some basic usage of the program can be found in `docs/demo.ttyrecord`. 
The [ttyplay](http://0xcc.net/ttyrec/index.html.en) tool is required to play the demo.

## License

The source code is licensed under the [MIT license](http://opensource.org/licenses/MIT).
