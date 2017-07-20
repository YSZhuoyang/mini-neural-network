
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    delete[] layerArr;
    layerArr = nullptr;
}


void NeuralNetwork::initLayers(
    const unsigned int numInstances,
    const unsigned int numLayers,
    const unsigned int* architecture )
{
    this->numLayers = numLayers;
    numHiddenLayers = numLayers - 1;
    layerArr = new Layer[numLayers];

    for (unsigned int i = 0; i < numLayers; i++)
    {
        unsigned short layerType;
        if (i == 0) layerType = FIRST_HIDDEN_LAYER;
        else if (i == numLayers - 1) layerType = OUTPUT_LAYER;
        else layerType = MIDDLE_HIDDEN_LAYER;

        layerArr[i].init(
            numInstances,
            architecture[i],
            architecture[i + 1],
            layerType );
        // printf( "layer: %d\n", i );
    }
}

void NeuralNetwork::train(
    const float* featureMat,
    const unsigned short* classIndexVec,
    const unsigned int maxIter,
    const float learningRate,
    const float costThreshold )
{
    unsigned int iter = 0;
    while (iter++ < maxIter)
    {
        activate( featureMat, classIndexVec );
        backProp();
    }
}

void NeuralNetwork::activate(
    const float* featureMat,
    const unsigned short* classIndexVec )
{
    // Forward propagation
    const float* inputMat = featureMat;
    for (unsigned int i = 0; i < numLayers; i++)
    {
        printf( "layer: %d forward output ...\n", i );
        inputMat = layerArr[i].forwardOutput( inputMat );
    }
    layerArr[numHiddenLayers].computeOutputLayerError( classIndexVec );
}

void NeuralNetwork::backProp()
{
    // Backword propagation
    for (unsigned int i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer: %d back propagate error ...\n", i );
        layerArr[i].backPropError(
            layerArr[i - 1].getErrorPtr(),
            layerArr[i - 1].getOutputPtr() );
    }
}
