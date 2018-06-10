

################################ Macros #################################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -O3 -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = Helper.o ArffImporter.o Layer.o Connection.o GradientDescent.o MiniNeuralNets.o Main.o

################################ Compile ################################

run: gpu_exec

gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

Helper.o: Helper.cpp Helper.h BasicDataStructures.h
	$(NVCC) ${NVCCCFLAGS} -c Helper.cpp

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} -c ArffImporter.cpp

Layer.o: Layer.cpp Layer.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Layer.cpp

Connection.o: Connection.cpp Connection.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Connection.cpp

GradientDescent.o: GradientDescent.cpp GradientDescent.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -c GradientDescent.cpp

MiniNeuralNets.o: MiniNeuralNets.cpp MiniNeuralNets.h BasicDataStructures.h GradientDescent.o Layer.o Connection.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c MiniNeuralNets.cpp

Main.o: Main.cpp MiniNeuralNets.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Main.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
