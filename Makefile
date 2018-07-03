

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++14 -O3 -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o ActivationFunction.o Sigmoid.o HyperTangent.o MiniNeuralNets.o GradientDescent.o Main.o

################################ Compile ################################

run: gpu_exec

gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

Helper.o: Helper.cpp Helper.hpp BasicDataStructures.hpp
	$(NVCC) ${NVCCCFLAGS} -c Helper.cpp

ArffImporter.o: ArffImporter.cpp ArffImporter.hpp Helper.o
	$(NVCC) ${NVCCCFLAGS} -c ArffImporter.cpp

ActivationFunction.o: ActivationFunction.cpp ActivationFunction.hpp
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c ActivationFunction.cpp

Sigmoid.o: Sigmoid.cpp Sigmoid.hpp ActivationFunction.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c Sigmoid.cpp

HyperTangent.o: HyperTangent.cpp HyperTangent.hpp ActivationFunction.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c HyperTangent.cpp

MiniNeuralNets.o: MiniNeuralNets.cpp MiniNeuralNets.hpp Layer.hpp Connection.hpp ActivationFunction.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c MiniNeuralNets.cpp

GradientDescent.o: GradientDescent.cpp GradientDescent.hpp MiniNeuralNets.o
	$(NVCC) ${NVCCCFLAGS} -c GradientDescent.cpp

Main.o: Main.cpp GradientDescent.o MiniNeuralNets.o Sigmoid.o HyperTangent.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c Main.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
