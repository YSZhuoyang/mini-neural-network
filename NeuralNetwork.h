
#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include "Layer.h"


class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void initLayers(
        const unsigned int numInstances,
        const unsigned int numLayers,
        // An array of length which equals to numLayers + 1
        // All except last count include bias
        const unsigned int* architecture,
        cublasHandle_t cublasHandle );
    void train(
        const float* featureMat,
        const unsigned short* classIndexVec,
        const unsigned int maxIter,
        const float learningRate,
        const float regularParam,
        const float costThreshold );

private:
    void forwardProp();
    void backProp(
        const float learningParam,
        const float regularParam );

    // To be deleted
    const unsigned short* classIndexVec = nullptr;
    // Does not include input layer
    float* dFeatureMat               = nullptr;
    unsigned short* dClassIndexVec   = nullptr;
    // Number of features in each layer including input layer
    const unsigned int* architecture = nullptr;
    Layer* layerArr                  = nullptr;
    unsigned int numInstances        = 0;
    // Number of layers excluding input layer
    unsigned short numLayers         = 0;
    unsigned short numHiddenLayers   = 0;
};

#endif
