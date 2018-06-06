
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

    struct LayerKernalConfig
    {
        dim3 sigBlockDim;
        dim3 sigGridDim;
        dim3 ccBlockDim;
        dim3 ccGridDim;
    };

    struct ConnectionKernalConfig
    {
        dim3 uwBlockDim;
        dim3 uwGridDim;
    };

    struct Layer
    {
        float* outputMat;
        float* dOutputMat;
        float* errorMat;
        float* dErrorMat;
        LayerKernalConfig layerKernalConfig;
        unsigned int numNodes;
        unsigned int outputMatSize;
        unsigned int errorMatSize;
        LayerType layerType;
    };

    struct Connection
    {
        float* weightMat;
        float* dWeightMat;
        float* deltaWeightMat;
        float* dDeltaWeightMat;
        unsigned int weightMatSize;
    };
}

#endif
