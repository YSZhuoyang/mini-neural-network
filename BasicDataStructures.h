
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_


namespace BasicDataStructures
{
    enum LayerType
    {
        INPUT_LAYER,
        HIDDEN_LAYER,
        OUTPUT_LAYER
    };

    struct Instance
    {
        float* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        float min;
        float max;
        float mean;
    };

    struct KernalConfig
    {
        dim3 blockDim;
        dim3 gridDim;
    };

    struct Layer
    {
        // Should we keep host data?
        float* outputMat;
        float* dOutputMat;
        float* errorMat;
        float* dErrorMat;
        KernalConfig sigKernalConfig;
        KernalConfig ccKernalConfig;
        // A node is a neuron processor
        unsigned int numNodes;
        // Includes output of all nodes and bias output which is always 1.0
        unsigned int numFeatures;
        unsigned int outputMatSize;
        unsigned int errorMatSize;
        LayerType layerType;
    };

    struct Connection
    {
        float* weightMat;
        float* dWeightMat;
        float* dDeltaWeightMat;
        KernalConfig uwKernalConfig;
        unsigned int weightMatSize;
        unsigned int numFeaturesIn;
        // Number of output features, which is equal to the number of nodes in the next layer
        unsigned int numFeaturesOut;
    };
}

#endif
