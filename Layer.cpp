
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
    if (layerType == OUTPUT_LAYER && numFeaturesOut == 2)
    {
        printf( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );
        return;
    }

    this->numInstances = numInstances;
    this->numFeaturesOut = numFeaturesOut;
    this->numFeaturesIn = numFeaturesIn;
    this->layerType = layerType;
    numNodes = (layerType == OUTPUT_LAYER) ?
        numFeaturesOut : numFeaturesOut - 1;

    weightMat = (float*) malloc( numFeaturesIn * numNodes * sizeof( float ) );
    outputMat = (float*) malloc( numInstances * numFeaturesOut * sizeof( float ) );
    errorMat = (float*) malloc( numInstances * numNodes * sizeof( float ) );

    // Inie weight matrix
    for (unsigned int i = 0; i < numNodes * numFeaturesIn; i++)
        weightMat[i] = ((float) (rand() % 101) - 50.0f) / 50.0f;
}

float* Layer::forwardOutput( const float* inputMat )
{
    // Include bias in non-output layer
    unsigned int offset = 1;
    if (layerType == OUTPUT_LAYER) offset = 0;
    else
        // Fill the first feature with X0 for bias
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
    // Ignore bias input
    unsigned int offset = 1;
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

    // float sum = 0.0f;
    // for (int i = 0; i < numNodesPreLayer; i++)
    //     for (int j = 0; j < numInstances; j++)
    //         sum += preLayerErrorMat[j * numNodesPreLayer + i];
    // printf( "Pre Error sum: %f\n", sum );

    // printf( "error in: %f\n", preLayerErrorMat[0] );
}

void Layer::updateWeights(
    const float* inputMat,
    const float learningParam,
    const float regularParam )
{
    for (unsigned int idNode = 0; idNode < numNodes; idNode++)
        for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int i = 0; i < numInstances; i++)
                sum += inputMat[numFeaturesIn * i + idIn] *
                    errorMat[numNodes * i + idNode];
            if (idIn != 0)
                sum += regularParam * weightMat[numFeaturesIn * idNode + idIn];
            weightMat[numFeaturesIn * idNode + idIn] += learningParam * sum;
        }

    // float sum = 0.0f;
    // for (int i = 0; i < numNodes; i++)
    //     for (int j = 0; j < numFeaturesIn; j++)
    //         sum += weightMat[i * numFeaturesIn + j];
    // printf( "Back propagate completed, Weight sum: %f\n", sum );
}

void Layer::computeOutputLayerError( const unsigned short* classIndexVec )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be ran by output layer.\n" );
        return;
    }

    // 2 classes
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
            costSum -= (classIndexVec[i]) ?
                logf(outputMat[i * numNodes + j]) : logf(1.0f - outputMat[i * numNodes + j]);
            // costSum += fabs(errorMat[i * numNodes + j]);

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
