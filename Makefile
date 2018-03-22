

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -O3 -use_fast_math -lcublas
# Enable host code debug in vscode
NVCCCFLAGS_DEBUG = -arch=sm_50 -std=c++11 -use_fast_math -lcublas -g
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o Layer.o NeuralNetwork.o Main.o

################################ Compile ################################

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

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
