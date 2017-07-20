#ifndef _LAYER_H_
#define _LAYER_H_

#include <cstring>
#include "Helper.h"


#define HIDDEN_LAYER 0
#define OUTPUT_LAYER 1

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
        const unsigned short layerType );
    float* forwardOutput( const float* inputMat );
    float* getOutputPtr();
    float* getErrorPtr();
    float* getWeightPtr();
    void backPropError(
        float* preLayerErrorMat,
        const float* inputMat );
    void updateWeights(
        const float* inputMat,
        const float learningRate );
    void computeOutputLayerError( const unsigned short* classIndexVec );


private:
    unsigned int numInstances   = 0;
    unsigned int numFeaturesOut = 0;
    unsigned int numFeaturesIn  = 0;
    unsigned int numNodes       = 0;
    unsigned int layerType      = 0;
    float* weightMat            = nullptr;
    float* outputMat            = nullptr;
    float* preLayerErrorMat     = nullptr;
    float* errorMat             = nullptr;
};

#endif
