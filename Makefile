

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -O3 -use_fast_math -lcublas
# Enable host code debug in vscode
NVCCCFLAGS_DEBUG = -arch=sm_50 -std=c++11 -g -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o Layer.o NeuralNetwork.o Main.o
OBJECTS_DEBUG = Helper_debug.o ArffImporter_debug.o Layer_debug.o NeuralNetwork_debug.o Main_debug.o

############################# Compile exec ##############################

run: gpu_exec

gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

Helper.o: Helper.cpp Helper.h BasicDataStructures.h
	$(NVCC) ${NVCCCFLAGS} -c Helper.cpp

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} -c ArffImporter.cpp

Layer.o: Layer.cpp Layer.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c Layer.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.h BasicDataStructures.h Layer.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c NeuralNetwork.cpp

Main.o: Main.cpp NeuralNetwork.o Layer.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Main.cpp

###################### Compile with debug enabled #######################

debug: gpu_exec_debug

gpu_exec_debug: ${OBJECTS_DEBUG}
	$(NVCC) ${NVCCCFLAGS_DEBUG} -o $@ ${OBJECTS_DEBUG}

Helper_debug.o: Helper.cpp Helper.h BasicDataStructures.h
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c Helper.cpp -o Helper_debug.o

ArffImporter_debug.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c ArffImporter.cpp -o ArffImporter_debug.o

Layer_debug.o: Layer.cpp Layer.h BasicDataStructures.h Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} ${CUFLAGS} -c Layer.cpp -o Layer_debug.o

NeuralNetwork_debug.o: NeuralNetwork.cpp NeuralNetwork.h BasicDataStructures.h Layer_debug.o Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c NeuralNetwork.cpp -o NeuralNetwork_debug.o

Main_debug.o: Main.cpp NeuralNetwork_debug.o Layer_debug.o Helper_debug.o
	$(NVCC) ${NVCCCFLAGS_DEBUG} -c Main.cpp -o Main_debug.o

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
