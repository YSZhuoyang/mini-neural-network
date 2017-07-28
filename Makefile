

################################ Macros #################################

SHELL = /bin/sh
CC = g++
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS = -Ofast -march=native -mtune=native -std=c++11
# OBJECTS = Helper.o ArffImporter.o Layer.o NeuralNetwork.o Main.o

# Enable Nvidia gpu
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++11 -O3 -use_fast_math -lcublas# -lcublas_device -rdc=true -lcudadevrt
OBJECTS = Helper.o ArffImporter.o Layer.o NeuralNetwork.o Main.o

################################ Compile ################################

# Compile with NVCC
gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.o
	$(CC) ${CFLAGS} -c ArffImporter.cpp

Layer.o: Layer.cpp Layer.h BasicDataStructures.h Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Layer.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.h BasicDataStructures.h Layer.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c NeuralNetwork.cpp

Main.o: Main.cpp NeuralNetwork.o Layer.o Helper.o
	$(NVCC) ${NVCCCFLAGS} -c Main.cpp

# Compile with GCC
# exec: ${OBJECTS}
# 	$(CC) ${CFLAGS} -o $@ ${OBJECTS}

# Helper.o: Helper.c Helper.h BasicDataStructures.h
# 	$(CC) ${CFLAGS} -c Helper.c

# ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.o
# 	$(CC) ${CFLAGS} -c ArffImporter.cpp

# Layer.o: Layer.cpp Layer.h BasicDataStructures.h Helper.o
# 	$(CC) ${CFLAGS} -c Layer.cpp

# NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.h BasicDataStructures.h Layer.o Helper.o
# 	$(CC) ${CFLAGS} -c NeuralNetwork.cpp

# Main.o: Main.cpp NeuralNetwork.o Layer.o Helper.o
# 	$(CC) ${CFLAGS} -c Main.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
