#ifndef _LAYER_H_
#define _LAYER_H_

#include <cstring>
#include <random>
#include "Helper.h"



#define NUM_BLOCK_THREADS 128

using namespace MyHelper;

class Layer
{
public:
    Layer();
    ~Layer();

    void init(
        const unsigned int numFeaturesIn,
        const unsigned int numFeaturesOut,
        // Determine wether X0 for bias is included in output features
        const LayerType layerType,
        cublasHandle_t cublasHandle );
    void initWeightData();
    void initOutputBuffers( const unsigned int numInstances );
    float* forwardOutput(
        const float* dInputMat,
        cudaStream_t stream );
    void backPropError(
        const float* dNextLayerErrorMat,
        const float* dNextLayerWeightMat,
        const unsigned int numNextLayerFeasOut,
        cudaStream_t stream );
    void computeOutputLayerError(
        const unsigned short* dClassIndexMat,
        cudaStream_t stream );
    void updateWeights(
        const float* dInputMat,
        const float learningParam,
        const float regularParam,
        cudaStream_t stream );
    float computeCost(
        float* dCostMat,
        const unsigned short* dClassIndexMat,
        cudaStream_t stream );
    float* getOutputPtr();
    float* getErrorPtr();
    float* getWeightPtr();
    float* getDWeightPtr();
    float* getDOutputPtr();
    float* getDErrorPtr();
    unsigned int getNumNodes();


private:
    // Host data
    float* weightMat            = nullptr;
    float* outputMat            = nullptr;
    float* errorMat             = nullptr;
    // Device data
    float* dWeightMat           = nullptr;
    float* dDeltaWeightMat      = nullptr;
    float* dOutputMat           = nullptr;
    float* dOutputMatOffset     = nullptr;
    float* dErrorMat            = nullptr;
    unsigned int numInstances   = 0;
    unsigned int numFeaturesOut = 0;
    unsigned int numFeaturesIn  = 0;
    unsigned int numNodes       = 0;
    unsigned int weightMatSize  = 0;
    unsigned int errorMatSize   = 0;
    unsigned int outputMatSize  = 0;
    LayerType layerType;
    // Kernel config
    dim3 sigBlockDim;
    dim3 sigGridDim;
    dim3 ccBlockDim;
    dim3 ccGridDim;
    dim3 uwBlockDim;
    dim3 uwGridDim;
    cublasHandle_t cublasHandle;
};

#endif
