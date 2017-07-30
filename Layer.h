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
    void backPropError(
        float* preLayerErrorMat,
        const float* dInputMat,
        dim3 bpeGridDim,
        dim3 bpeBlockDim );
    void updateWeights(
        const float* dInputMat,
        const float learningParam );
    void computeOutputLayerError(
        const unsigned short* dClassIndexVec,
        const unsigned short* classIndexVec );
    float* getOutputPtr();
    float* getErrorPtr();
    float* getWeightPtr();
    float* getDWeightPtr();
    float* getDOutputPtr();
    float* getDErrorPtr();
    dim3 getSigGridDim();
    dim3 getSigBlockDim();


private:
    // Host data
    float* weightMat            = nullptr;
    float* outputMat            = nullptr;
    float* errorMat             = nullptr;
    float* preLayerErrorMat     = nullptr;
    // Device data
    float* dWeightMat           = nullptr;
    float* dDeltaWeightMat      = nullptr;
    float* dOutputMat           = nullptr;
    float* dOutputMatOffset     = nullptr;
    float* dErrorMat            = nullptr;
    float* dPreLayerErrorMat    = nullptr;
    unsigned int numInstances   = 0;
    unsigned int numFeaturesOut = 0;
    unsigned int numFeaturesIn  = 0;
    unsigned int numNodes       = 0;
    unsigned int outputOffset   = 0;
    unsigned int weightMatSize  = 0;
    unsigned int errorMatSize   = 0;
    unsigned int outputMatSize  = 0;
    unsigned int inputMatSize   = 0;
    unsigned int layerType      = 0;
    // Kernel config
    dim3 sigBlockDim;
    dim3 sigGridDim;
    dim3 ccBlockDim;
    dim3 ccGridDim;
    // dim3 bpeBlockDim;
    // dim3 bpeGridDim;
    cublasHandle_t cublasHandle;
};

#endif
