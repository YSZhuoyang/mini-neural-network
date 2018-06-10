
#include "Layer.h"
#include "Connection.h"
#include "GradientDescent.h"

#ifndef _MINI_NEURAL_NETS_H_
#define _MINI_NEURAL_NETS_H_


class MiniNeuralNets
{
public:
    MiniNeuralNets();
    ~MiniNeuralNets();

    void initialize(
        // An array of length which equals to numLayers + 1, and
        // all layers except output layer include bias input X0
        const std::vector<unsigned int>& architecture,
        cublasHandle_t cublasHandle );
    void train(
        const float* featureMat,
        const unsigned short* classIndexMat,
        const unsigned int numInstances,
        const unsigned int maxIter,
        const float learningRate,
        const float regularParam,
        const float costThreshold );
    void test(
        const float* featureMat,
        const unsigned short* classIndexMat,
        const unsigned int numInstances );

private:
    inline void forwardProp(
        const unsigned int numInstances,
        cudaStream_t stream1 );
    inline void backwardProp(
        const unsigned short* dClassIndexMat,
        const unsigned int numInstances,
        const float learningParam,
        const float regularParam,
        cudaStream_t stream1,
        cudaStream_t stream2 );

    // Number of features in each layer including input layer
    unsigned int* architecture       = nullptr;
    // Does not include input layer
    Layer* layers                    = nullptr;
    Connection* connections          = nullptr;
    // Number of layers excluding input layer
    unsigned short numLayers         = 0;
    unsigned short numHiddenLayers   = 0;
    unsigned short numConnections    = 0;

    cudaEvent_t* backPropCompletes   = nullptr;
    cudaEvent_t forwardPropComplete;
    cudaEvent_t trainingComplete;
    cudaEvent_t testComplete;
    cublasHandle_t cublasHandle;
};

#endif
