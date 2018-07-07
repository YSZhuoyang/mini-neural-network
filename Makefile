

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++14 -O3 -use_fast_math -lcublas
# Enable host code debug in vscode
NVCCCFLAGS_DEBUG = -arch=sm_50 -std=c++14 -g -G -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o ActivationFunction.o Sigmoid.o HyperTangent.o MiniNeuralNets.o GradientDescent.o Main.o
OBJECTS_DEBUG = Helper_debug.o ArffImporter_debug.o ActivationFunction_debug.o Sigmoid_debug.o HyperTangent_debug.o MiniNeuralNets_debug.o GradientDescent_debug.o Main_debug.o

############################# Compile exec ##############################

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

###################### Compile with debug enabled #######################

debug: gpu_exec_debug

gpu_exec_debug: ${OBJECTS_DEBUG}
	$(NVCC) ${NVCCCFLAGS_DEBUG} -o $@ ${OBJECTS_DEBUG}

Helper_debug.o: Helper.cpp Helper.hpp BasicDataStructures.hpp
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c Helper.cpp -o Helper_debug.o

ArffImporter_debug.o: ArffImporter.cpp ArffImporter.hpp Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c ArffImporter.cpp -o ArffImporter_debug.o

ActivationFunction_debug.o: ActivationFunction.cpp ActivationFunction.hpp
	$(NVCC) ${NVCCCFLAGS_DEBUG} ${CUFLAGS} -c ActivationFunction.cpp -o ActivationFunction_debug.o

Sigmoid_debug.o: Sigmoid.cpp Sigmoid.hpp ActivationFunction_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} ${CUFLAGS} -c Sigmoid.cpp -o Sigmoid_debug.o

HyperTangent_debug.o: HyperTangent.cpp HyperTangent.hpp ActivationFunction_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} ${CUFLAGS} -c HyperTangent.cpp -o HyperTangent_debug.o

MiniNeuralNets_debug.o: MiniNeuralNets.cpp MiniNeuralNets.hpp Layer.hpp Connection.hpp ActivationFunction_debug.o Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c MiniNeuralNets.cpp -o MiniNeuralNets_debug.o

GradientDescent_debug.o: GradientDescent.cpp GradientDescent.hpp MiniNeuralNets_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c GradientDescent.cpp -o GradientDescent_debug.o

Main_debug.o: Main.cpp GradientDescent_debug.o MiniNeuralNets_debug.o Sigmoid_debug.o HyperTangent_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} ${CUFLAGS} -c Main.cpp -o Main_debug.o

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
