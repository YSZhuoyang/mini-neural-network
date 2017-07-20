
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
        if (i == numLayers - 1) layerType = OUTPUT_LAYER;
        else layerType = HIDDEN_LAYER;

        layerArr[i].init(
            numInstances,
            architecture[i],
            architecture[i + 1],
            layerType );
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
        backProp( featureMat, learningRate );

        printf( "\n" );
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
        // printf( "output: %f\n", inputMat[0] );
    }
    layerArr[numHiddenLayers].computeOutputLayerError( classIndexVec );
}

void NeuralNetwork::backProp(
    const float* featureMat,
    const float learningRate )
{
    // Backword propagation
    for (unsigned int i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer: %d back propagate ...\n", i );
        layerArr[i].backPropError(
            layerArr[i - 1].getErrorPtr(),
            layerArr[i - 1].getOutputPtr() );
        printf( "layer: %d update weights ...\n", i );
        layerArr[i].updateWeights(
            layerArr[i - 1].getOutputPtr(),
            learningRate );
        printf( "Weight: %f\n", layerArr[i].getWeightPtr()[0] );
    }

    printf( "layer: 0 update weights ...\n" );
    layerArr[0].updateWeights(
        featureMat,
        learningRate );
    printf( "Weight: %f\n", layerArr[0].getWeightPtr()[0] );

    // float sum = 0.0f;
    // for (int i = 0; i < 11; i++)
    //     for (int j = 0; j < 1001; j++)
    //         sum += layerArr[0].getWeight()[i * 1001 + j];
    // printf( "Weight sum: %f\n", sum );
}
