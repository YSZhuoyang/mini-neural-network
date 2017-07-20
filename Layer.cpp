
#include "Layer.h"


Layer::Layer()
{

}

Layer::~Layer()
{
    free( weightMat );
    free( outputMat );
    free( errorMat );
    weightMat = nullptr;
    outputMat = nullptr;
    errorMat = nullptr;
}


void Layer::init(
    const unsigned int numInstances,
    const unsigned int numFeaturesIn,
    const unsigned int numFeaturesOut,
    const unsigned short layerType )
{
    this->numInstances = numInstances;
    this->numFeaturesOut = numFeaturesOut;
    this->numFeaturesIn = numFeaturesIn;
    this->layerType = layerType;
    numNodes = (layerType == OUTPUT_LAYER) ?
        numFeaturesOut : numFeaturesOut - 1;

    weightMat = (float*) calloc( numFeaturesIn * numNodes, sizeof( float ) );
    outputMat = (float*) malloc( numInstances * numFeaturesOut * sizeof( float ) );
    errorMat = (float*) malloc( numInstances * numNodes * sizeof( float ) );
}

float* Layer::forwardOutput( const float* inputMat )
{
    // Include bias in non-output layer
    unsigned int offset = 1;
    if (layerType == OUTPUT_LAYER) offset = 0;
    else
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i * numFeaturesOut] = 1;

    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int idNode = 0; idNode < numNodes; idNode++)
        {
            float sum = 0.0f;
            for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
                sum += weightMat[idNode * numFeaturesIn + idIn] *
                    inputMat[i * numFeaturesIn + idIn];
            sum = 1.0f / (1.0f + expf(-sum));
            outputMat[numFeaturesOut * i + idNode + offset] = sum;
        }

    return outputMat;
}

void Layer::backPropError(
    float* preLayerErrorMat,
    const float* inputMat )
{
    unsigned int numNodesPreLayer = numFeaturesIn - 1;
    unsigned int offset = 1;
    // Do not consider bias
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int idIn = 0; idIn < numNodesPreLayer; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int idNode = 0; idNode < numNodes; idNode++)
                sum += weightMat[idNode * numFeaturesIn + idIn + offset] *
                    errorMat[numNodes * i + idNode];
            preLayerErrorMat[numNodesPreLayer * i + idIn] =
                sum * inputMat[numFeaturesIn * i + idIn + offset] *
                (1.0f - inputMat[numFeaturesIn * i + idIn + offset]);
        }

    // printf( "error in: %f\n", preLayerErrorMat[0] );
}

void Layer::updateWeights(
    const float* inputMat,
    const float learningRate )
{
    for (unsigned int idNode = 0; idNode < numNodes; idNode++)
        for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int i = 0; i < numInstances; i++)
                sum += inputMat[numFeaturesIn * i + idIn] *
                    errorMat[numNodes * i + idNode];
            weightMat[numFeaturesIn * idNode + idIn] -=
                learningRate / (float) numInstances * sum;
        }

    printf( "Back propagate completed, weight: %f\n", weightMat[0] );
}

void Layer::computeOutputLayerError( const unsigned short* classIndexVec )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be ran by output layer.\n" );
        return;
    }

    // Assume there are 2 classes
    if (numFeaturesOut == 1)
        for (unsigned int i = 0; i < numInstances; i++)
            errorMat[i] = outputMat[i] - (float) classIndexVec[i];
    // More than 2 classes
    else
    {
        memmove( errorMat, outputMat, numInstances * numNodes * sizeof( float ) );
        for (unsigned int i = 0; i < numInstances; i++)
            errorMat[i * numNodes + classIndexVec[i]] -= 1.0f;
    }

    float costSum = 0.0f;
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int j = 0; j < numNodes; j++)
            // costSum -= (classIndexVec[i]) ?
            //     logf(outputMat[i * numNodes + j]) : logf(1.0f - outputMat[i * numNodes + j]);
            costSum += fabs(errorMat[i * numNodes + j]);

    printf( "Cost: %f\n", costSum );
}

float* Layer::getWeightPtr()
{
    return weightMat;
}

float* Layer::getOutputPtr()
{
    return outputMat;
}

float* Layer::getErrorPtr()
{
    return errorMat;
}
