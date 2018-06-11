

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -O3 -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o GradientDescent.o MiniNeuralNets.o Main.o

################################ Compile ################################

run: gpu_exec

gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

Helper.o: Helper.cpp Helper.hpp BasicDataStructures.hpp
	$(NVCC) ${NVCCCFLAGS} -c Helper.cpp

ArffImporter.o: ArffImporter.cpp ArffImporter.hpp Helper.o
	$(NVCC) ${NVCCCFLAGS} -c ArffImporter.cpp

MiniNeuralNets.o: MiniNeuralNets.cpp MiniNeuralNets.hpp Layer.hpp Connection.hpp Helper.o
	$(NVCC) ${NVCCCFLAGS} -c MiniNeuralNets.cpp

GradientDescent.o: GradientDescent.cpp GradientDescent.hpp ActivationFunction.hpp MiniNeuralNets.o
	$(NVCC) ${NVCCCFLAGS} -c GradientDescent.cpp

Main.o: Main.cpp GradientDescent.o MiniNeuralNets.o Sigmoid.hpp
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c Main.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
