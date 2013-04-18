Dwarf Mine - The 13-11 Benchmark
================================

This was a Master's project by the [Operating Systems and Middleware Group](http://www.dcl.hpi.uni-potsdam.de) at the [Hasso-Plattner-Institute (HPI)](http://www.hpi-web.de) in Potsdam, Germany.  
Our goal was to define and implement a **parallel benchmark suite for heterogeneous many-core systems**. 

## Overview

## Build instructions

This is a CMake based project, but it has only been compiled and tested in Linux environments. In order to build and run the GPU implementations, a Fermi-class NVIDIA graphics card is required.
Setting the `BUILD_WITH_CUDA` CMake option to `OFF` will disable building the CUDA implementations.

### Prerequisites

* GCC 4.7 or higher
* CUDA 5.0 or higher (optional)
* OpenMPI 1.6.4 or higher, with multithreading support
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

## License

The source code is licensed under the [MIT license](http://opensource.org/licenses/MIT). Also see the file `LICENSE`.
