

################################# Paths #################################

BUILDDIR = build
BINDIR = bin

MODELHEADERDIR = include/model
ACTHEADERDIR = include/act
TRAINERHEADERDIR = include/trainer
UTILHEADERDIR = include/util

MODELSRCDIR = src/model
ACTSRCDIR = src/act
TRAINERSRCDIR = src/trainer
UTILSRCDIR = src/util

IINC = include/
ILIBCUTLASS = lib/cutlass

############################# Build options #############################

SHELL = /bin/sh
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
	nvcc ${NVCCCFLAGS} -o $@ ${OBJECTS}

$(BUILDDIR)/Helper.o: $(UTILSRCDIR)/Helper.cpp $(UTILHEADERDIR)/Helper.hpp $(MODELHEADERDIR)/BasicDataStructures.hpp
	nvcc ${NVCCCFLAGS} -I $(IINC) -c $(UTILSRCDIR)/Helper.cpp -o $(BUILDDIR)/Helper.o

$(BUILDDIR)/ArffImporter.o: $(UTILSRCDIR)/ArffImporter.cpp $(UTILHEADERDIR)/ArffImporter.hpp $(BUILDDIR)/Helper.o
	nvcc ${NVCCCFLAGS} -I $(IINC) -c $(UTILSRCDIR)/ArffImporter.cpp -o $(BUILDDIR)/ArffImporter.o

$(BUILDDIR)/Sigmoid.o: $(ACTSRCDIR)/Sigmoid.cpp $(ACTHEADERDIR)/Sigmoid.hpp $(MODELHEADERDIR)/Layer.hpp $(MODELHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c $(ACTSRCDIR)/Sigmoid.cpp -o $(BUILDDIR)/Sigmoid.o

$(BUILDDIR)/HyperTangent.o: $(ACTSRCDIR)/HyperTangent.cpp $(ACTHEADERDIR)/HyperTangent.hpp $(MODELHEADERDIR)/Layer.hpp $(MODELHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c $(ACTSRCDIR)/HyperTangent.cpp -o $(BUILDDIR)/HyperTangent.o

$(BUILDDIR)/Relu.o: $(ACTSRCDIR)/Relu.cpp $(ACTHEADERDIR)/Relu.hpp $(MODELHEADERDIR)/Layer.hpp $(MODELHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c $(ACTSRCDIR)/Relu.cpp -o $(BUILDDIR)/Relu.o

$(BUILDDIR)/MiniNeuralNets.o: $(MODELSRCDIR)/MiniNeuralNets.cpp $(MODELHEADERDIR)/MiniNeuralNets.hpp $(MODELHEADERDIR)/Layer.hpp $(MODELHEADERDIR)/Connection.hpp $(ACTHEADERDIR)/ActivationFunction.hpp $(BUILDDIR)/Helper.o
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c $(MODELSRCDIR)/MiniNeuralNets.cpp -o $(BUILDDIR)/MiniNeuralNets.o

$(BUILDDIR)/GradientDescent.o: $(TRAINERSRCDIR)/GradientDescent.cpp $(TRAINERHEADERDIR)/GradientDescent.hpp $(BUILDDIR)/MiniNeuralNets.o
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c $(TRAINERSRCDIR)/GradientDescent.cpp -o $(BUILDDIR)/GradientDescent.o

$(BUILDDIR)/Main.o: src/Main.cpp $(BUILDDIR)/GradientDescent.o $(BUILDDIR)/MiniNeuralNets.o $(BUILDDIR)/Sigmoid.o $(BUILDDIR)/HyperTangent.o
	nvcc ${NVCCCFLAGS} ${CUFLAGS} -I $(IINC) -I $(ILIBCUTLASS) -c src/Main.cpp -o $(BUILDDIR)/Main.o

################################# Clean #################################

clean:
	-rm -f $(BUILDDIR)/*.o $(BINDIR)/*exec*
