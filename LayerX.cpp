
#include "Helper.h"

using namespace BasicDataStructures;
using namespace MyHelper;

#define NUM_BLOCK_THREADS 128

Layer initializeLayer(
    const unsigned int numFeaturesIn,
    const unsigned int numFeaturesOut,
    const unsigned int numInstances,
    const LayerType layerType)
{
    if (layerType == OUTPUT_LAYER && numFeaturesOut == 2)
    {
        printf( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );
        return;
    }

    unsigned int numNodes =
        (layerType == OUTPUT_LAYER) ? numFeaturesOut : numFeaturesOut - 1;
    unsigned int outputMatSize = numFeaturesOut * numInstances;
    unsigned int errorMatSize = numNodes * numInstances;

    /* Determine block and grid size of kernel functions */
    dim3 ccBlockDim;
    dim3 ccGridDim;
    dim3 sigBlockDim;
    dim3 sigGridDim;
    if (outputMatSize > NUM_BLOCK_THREADS)
    {
        ccBlockDim.x = NUM_BLOCK_THREADS;
        ccGridDim.x = (outputMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    }
    else ccBlockDim.x = outputMatSize;

    if (errorMatSize > NUM_BLOCK_THREADS)
    {
        sigBlockDim.x = NUM_BLOCK_THREADS;
        sigGridDim.x = (errorMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    }
    else sigBlockDim.x = errorMatSize;


    LayerKernalConfig layerKernalConfig;
    layerKernalConfig.ccBlockDim = ccBlockDim;
    layerKernalConfig.ccGridDim = ccGridDim;
    layerKernalConfig.sigBlockDim = sigBlockDim;
    layerKernalConfig.sigGridDim = sigGridDim;

    /* Init host buffer data */
    float* errorMat = (float*) malloc( errorMatSize * sizeof( float ) );
    float* outputMat = (float*) malloc( outputMatSize * sizeof( float ) );

    /* Init device buffer data */
    float* dOutputMat;
    float* dErrorMat;
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, outputMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, errorMatSize * sizeof( float ) ) );
    // float* dOutputMatOffset = (layerType != HIDDEN_LAYER) ? dOutputMat : dOutputMat + numInstances;

    // Setup bias in non-output layer
    if (layerType == HIDDEN_LAYER)
    {
        // Fill the first feature with X0 for bias
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i] = 1.0f;
        
        // Fill in with X0 as bias
        cudaErrorCheck( cudaMemcpyAsync(
            dOutputMat,
            outputMat,
            numInstances * sizeof( float ),
            cudaMemcpyHostToDevice ) );
    }

    Layer layer;
    layer.layerType = layerType;
    layer.layerKernalConfig = layerKernalConfig;
    layer.outputMat = outputMat;
    layer.dOutputMat = dOutputMat;
    layer.errorMat = errorMat;
    layer.dErrorMat = dErrorMat;
    layer.numNodes = numNodes;
    layer.outputMatSize = outputMatSize;
    layer.errorMatSize = errorMatSize;

    return layer;

    // // this->numFeaturesOut = numFeaturesOut;
    // // this->numFeaturesIn = numFeaturesIn;
    // // weightMatSize = numFeaturesIn * numNodes;

    // // Allocate host memo
    // weightMat = (float*) malloc( weightMatSize * sizeof( float ) );
    // /* Compute block and grid size of kernel functions */
    // if (weightMatSize > NUM_BLOCK_THREADS)
    // {
    //     uwBlockDim.x = NUM_BLOCK_THREADS;
    //     uwGridDim.x = (weightMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    // }
    // else uwBlockDim.x = weightMatSize;

    // // Allocate device memo
    // cudaErrorCheck( cudaMalloc( (void**) &dWeightMat, weightMatSize * sizeof( float ) ) );
    // cudaErrorCheck( cudaMalloc( (void**) &dDeltaWeightMat, weightMatSize * sizeof( float ) ) );
}

void destroyLayer(Layer* layer)
{
    free( layer->outputMat );
    free( layer->errorMat );
    cudaErrorCheck( cudaFree( layer->dOutputMat ) );
    cudaErrorCheck( cudaFree( layer->dErrorMat ) );
}
