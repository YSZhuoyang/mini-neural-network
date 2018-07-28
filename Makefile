

################################# Paths #################################

BUILDDIR = build
BINDIR = bin

DATAHEADERDIR = include/datastruct
ACTHEADERDIR = include/act
TRAINERHEADERDIR = include/trainer
UTILHEADERDIR = include/util

DATASRCDIR = src/datastruct
ACTSRCDIR = src/act
TRAINERSRCDIR = src/trainer
UTILSRCDIR = src/util

INCACTSRC = src/act/../..
INCDATASRC = src/datastruct/../..
INCTRAINERSRC = src/trainer/../..
INCUTILSRC = src/util/../../

INCACTHEADER = include/act/../..
INCDATAHEADER = include/datastruct/../..
INCTRAINERHEADER = include/trainer/../..
INCUTILHEADER = include/util/../../

############################# Build options #############################

SHELL = /bin/sh
NVCC = nvcc
NVCCCFLAGS = -arch=sm_50 -std=c++14 -O3 -use_fast_math -lcublas
CUFLAGS = -x cu
OBJECTS = $(BUILDDIR)/Helper.o $(BUILDDIR)/ArffImporter.o $(BUILDDIR)/Sigmoid.o $(BUILDDIR)/HyperTangent.o
OBJECTS += $(BUILDDIR)/Relu.o $(BUILDDIR)/MiniNeuralNets.o $(BUILDDIR)/GradientDescent.o $(BUILDDIR)/Main.o

############################# Compile exec ##############################

run: $(BINDIR)/gpu_exec

# Enable host code debug in vscode
debug: NVCCCFLAGS = -arch=sm_50 -std=c++14 -g -G -use_fast_math -lcublas
debug: $(BINDIR)/gpu_exec

$(BINDIR)/gpu_exec: ${OBJECTS}
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS}

$(BUILDDIR)/Helper.o: $(UTILSRCDIR)/Helper.cpp $(UTILHEADERDIR)/Helper.hpp $(DATAHEADERDIR)/BasicDataStructures.hpp
	$(NVCC) ${NVCCCFLAGS} -I $(INCUTILSRC) -c $(UTILSRCDIR)/Helper.cpp -o $(BUILDDIR)/Helper.o

$(BUILDDIR)/ArffImporter.o: $(UTILSRCDIR)/ArffImporter.cpp $(UTILHEADERDIR)/ArffImporter.hpp $(BUILDDIR)/Helper.o
	$(NVCC) ${NVCCCFLAGS} -I $(INCUTILSRC) -c $(UTILSRCDIR)/ArffImporter.cpp -o $(BUILDDIR)/ArffImporter.o

$(BUILDDIR)/Sigmoid.o: $(ACTSRCDIR)/Sigmoid.cpp $(ACTHEADERDIR)/Sigmoid.hpp $(DATAHEADERDIR)/Layer.hpp $(DATAHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -I $(INCACTSRC) -c $(ACTSRCDIR)/Sigmoid.cpp -o $(BUILDDIR)/Sigmoid.o

$(BUILDDIR)/HyperTangent.o: $(ACTSRCDIR)/HyperTangent.cpp $(ACTHEADERDIR)/HyperTangent.hpp $(DATAHEADERDIR)/Layer.hpp $(DATAHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -I $(INCACTSRC) -c $(ACTSRCDIR)/HyperTangent.cpp -o $(BUILDDIR)/HyperTangent.o

$(BUILDDIR)/Relu.o: $(ACTSRCDIR)/Relu.cpp $(ACTHEADERDIR)/Relu.hpp $(DATAHEADERDIR)/Layer.hpp $(DATAHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -I $(INCACTSRC) -c $(ACTSRCDIR)/Relu.cpp -o $(BUILDDIR)/Relu.o

$(BUILDDIR)/MiniNeuralNets.o: $(DATASRCDIR)/MiniNeuralNets.cpp $(DATAHEADERDIR)/MiniNeuralNets.hpp $(DATAHEADERDIR)/Layer.hpp $(DATAHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp $(BUILDDIR)/Helper.o
	$(NVCC) ${NVCCCFLAGS} -I $(INCDATASRC) -c $(DATASRCDIR)/MiniNeuralNets.cpp -o $(BUILDDIR)/MiniNeuralNets.o

$(BUILDDIR)/GradientDescent.o: $(TRAINERSRCDIR)/GradientDescent.cpp $(TRAINERHEADERDIR)/GradientDescent.hpp $(BUILDDIR)/MiniNeuralNets.o
	$(NVCC) ${NVCCCFLAGS} ${CUFLAGS} -I $(INCTRAINERSRC) -c $(TRAINERSRCDIR)/GradientDescent.cpp -o $(BUILDDIR)/GradientDescent.o

$(BUILDDIR)/Main.o: src/Main.cpp $(BUILDDIR)/GradientDescent.o $(BUILDDIR)/MiniNeuralNets.o $(BUILDDIR)/Sigmoid.o $(BUILDDIR)/HyperTangent.o
	$(NVCC) ${NVCCCFLAGS} -I src/.. -c src/Main.cpp -o $(BUILDDIR)/Main.o

################################# Clean #################################

clean:
	-rm -f $(BUILDDIR)/*.o $(BINDIR)/*exec*
