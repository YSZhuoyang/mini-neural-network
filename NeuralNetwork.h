
#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include "Layer.h"


class NeuralNetwork
{
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void initLayers(
        // An array of length which equals to numLayers + 1
        // All except last count include bias
        const unsigned int* architecture,
        const unsigned int numLayers,
        cublasHandle_t cublasHandle );
    void train(
        const float* featureMat,
        const unsigned short* classIndexMat,
        const unsigned int numInstances,
        const unsigned int maxIter,
        const float learningRate,
        const float regularParam,
        const float initialWeightRange,
        const float costThreshold );
    void test(
        const float* featureMat,
        const unsigned short* classIndexMat,
        const unsigned int numInstances );

private:
    inline void forwardProp(
        const float* dFeatureMat,
        const unsigned short* dClassIndexMat,
        cudaStream_t stream );
    inline void backProp(
        const float* dFeatureMat,
        const float learningParam,
        const float regularParam,
        cudaStream_t stream1,
        cudaStream_t stream2 );

    // Number of features in each layer including input layer
    const unsigned int* architecture = nullptr;
    // Does not include input layer
    Layer* layerArr                  = nullptr;
    // Number of layers excluding input layer
    unsigned short numLayers         = 0;
    unsigned short numHiddenLayers   = 0;

    cudaEvent_t* backPropCompletes   = nullptr;
    cudaEvent_t forwardPropComplete;
    cudaEvent_t trainingComplete;
    cudaEvent_t testComplete;
    cublasHandle_t cublasHandle;
};

#endif
