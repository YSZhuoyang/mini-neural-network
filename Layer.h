#ifndef _LAYER_H_
#define _LAYER_H_

#include <cstring>
#include "Helper.h"


#define HIDDEN_LAYER 0
#define OUTPUT_LAYER 1

using namespace MyHelper;

class Layer
{
public:
    Layer();
    ~Layer();

    void init(
        const unsigned int numInstances,
        const unsigned int numFeaturesIn,
        const unsigned int numFeaturesOut,
        // Determine wether X0 for bias is included in output features
        const unsigned short layerType,
        cublasHandle_t cublasHandle );
    float* forwardOutput( const float* dInputMat );
    float* getOutputPtr();
    float* getErrorPtr();
    float* getWeightPtr();
    float* getDWeightPtr();
    float* getDOutputPtr();
    float* getDErrorPtr();
    void backPropError(
        float* preLayerErrorMat,
        const float* inputMat );
    void updateWeights(
        const float* inputMat,
        const float learningRate );
    void computeOutputLayerError(
        const unsigned short* __restrict__ dClassIndexVec,
        const unsigned short* __restrict__ classIndexVec );


private:
    unsigned int numInstances   = 0;
    unsigned int numFeaturesOut = 0;
    unsigned int numFeaturesIn  = 0;
    unsigned int numNodes       = 0;
    unsigned int outputOffset   = 0;
    unsigned int layerType      = 0;
    // Host data
    float* weightMat            = nullptr;
    float* outputMat            = nullptr;
    float* errorMat             = nullptr;
    float* preLayerErrorMat     = nullptr;
    // Device data
    float* dWeightMat           = nullptr;
    float* dOutputMat           = nullptr;
    float* dOutputMatOffset     = nullptr;
    float* dErrorMat            = nullptr;
    float* dPreLayerErrorMat    = nullptr;
    // Kernel config
    dim3 sigBlockDim;
    dim3 sigGridDim;
    dim3 ccBlockDim;
    dim3 ccGridDim;
    cublasHandle_t cublasHandle;
};

#endif
