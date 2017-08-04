
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
        const unsigned int* architecture );
    void train(
        const float* featureMat,
        const unsigned short* classIndexVec,
        const unsigned int maxIter,
        const float learningRate,
        const float regularParam,
        const float costThreshold );

private:
    void forwardProp(
        const float* featureMat,
        const unsigned short* classIndexVec );
    void backProp(
        const float* featureMat,
        const float learningParam,
        const float regularParam );

    // Does not include input layer
    Layer* layerArr                = nullptr;
    unsigned int numInstances      = 0;
    unsigned short numLayers       = 0;
    unsigned short numHiddenLayers = 0;
};

#endif
