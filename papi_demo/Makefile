all: cpu gpu
cpu: cpu.cpp
	g++ -O3 -pthread -std=c++11 cpu.cpp -lpapi -o cpu
gpu: gpu.cu
	nvcc gpu.cu -O3 -lpapi -o gpu
.PHONY: clean
clean:
	rm -rf ./gpu ./cpu
