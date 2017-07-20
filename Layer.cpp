
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

    printf( "deleted\n" );
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

    weightMat = (float*) calloc( numFeaturesIn * numFeaturesOut, sizeof( float ) );
    outputMat = (float*) malloc( numInstances * numFeaturesOut * sizeof( float ) );
    errorMat = (float*) malloc( numInstances * numFeaturesOut * sizeof( float ) );
}

float* Layer::forwardOutput( const float* inputMat )
{
    unsigned int idOutStart = 1;
    if (layerType == OUTPUT_LAYER) idOutStart = 0;
    else
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i * numFeaturesOut] = 1;

    // printf( "test 1\n" );
    
    for (unsigned int i = 0; i < numInstances; i++)
    {
        for (unsigned int idOut = idOutStart; idOut < numFeaturesOut; idOut++)
        {
            float sum = 0.0f;
            for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
                sum += weightMat[idOut * numFeaturesIn + idIn] *
                    inputMat[i * numFeaturesIn + idIn];
            outputMat[numFeaturesOut * i + idOut] = sum;
        }
    }

    return outputMat;
}

void Layer::backPropError( float* preLayerErrorMat, const float* inputMat )
{
    if (layerType == FIRST_HIDDEN_LAYER)
    {
        printf( "backPropError() can only be ran by non-input layer.\n" );
        return;
    }

    for (unsigned int i = 0; i < numInstances; i++)
    {
        for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int idOut = 0; idOut < numFeaturesOut; idOut++)
                sum += weightMat[idOut * numFeaturesIn + idIn] *
                    errorMat[numFeaturesOut * i + idOut];
            preLayerErrorMat[numFeaturesIn * i + idIn] =
                sum * inputMat[numFeaturesIn * i + idIn] *
                (1.0f - inputMat[numFeaturesIn * i + idIn]);
        }
    }
}

void Layer::computeOutputLayerError( const unsigned short* classIndexVec )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be ran by output layer.\n" );
        return;
    }

    memmove( errorMat, outputMat, numInstances * numFeaturesOut * sizeof( float ) );
    for (unsigned int i = 0; i < numInstances; i++)
        errorMat[i * numFeaturesOut + classIndexVec[i]] -= 1.0f;
}

float* Layer::getOutputPtr()
{
    return outputMat;
}

float* Layer::getErrorPtr()
{
    return errorMat;
}
