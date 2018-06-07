
#include "Layer.h"


Layer initializeLayer(
    const unsigned int numFeatures,
    const unsigned int numInstances,
    const LayerType layerType)
{
    if (layerType == OUTPUT_LAYER && numFeatures == 2)
        throw( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );

    unsigned int numNodes =
        (layerType == OUTPUT_LAYER) ? numFeatures : numFeatures - 1;
    unsigned int outputMatSize = numFeatures * numInstances;
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

    KernalConfig ccKernalConfig;
    ccKernalConfig.blockDim = ccBlockDim;
    ccKernalConfig.gridDim = ccGridDim;
    KernalConfig sigKernalConfig;
    sigKernalConfig.blockDim = sigBlockDim;
    sigKernalConfig.gridDim = sigGridDim;

    /* Init host buffer data */
    float* errorMat = (float*) malloc( errorMatSize * sizeof( float ) );
    float* outputMat = (float*) malloc( outputMatSize * sizeof( float ) );

    /* Init device buffer data */
    float* dOutputMat;
    float* dErrorMat;
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, outputMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, errorMatSize * sizeof( float ) ) );

    // Setup bias in non-output layer
    if (layerType == HIDDEN_LAYER)
    {
        const unsigned int biasInputOffset = numInstances * numNodes;
        float* outputMatOffset = outputMat + biasInputOffset;
        float* dOutputMatOffset = dOutputMat + biasInputOffset;
        // Fill the first feature with X0 for bias
        for (unsigned int i = 0; i < numInstances; i++)
            outputMatOffset[i] = 1.0f;
        
        // Fill in with X0 as bias
        cudaErrorCheck( cudaMemcpyAsync(
            dOutputMatOffset,
            outputMatOffset,
            numInstances * sizeof( float ),
            cudaMemcpyHostToDevice ) );
    }

    Layer layer;
    layer.layerType = layerType;
    layer.sigKernalConfig = sigKernalConfig;
    layer.ccKernalConfig = ccKernalConfig;
    layer.outputMat = outputMat;
    layer.dOutputMat = dOutputMat;
    layer.errorMat = errorMat;
    layer.dErrorMat = dErrorMat;
    layer.numNodes = numNodes;
    layer.numFeatures = numFeatures;
    layer.outputMatSize = outputMatSize;
    layer.errorMatSize = errorMatSize;

    return layer;
}

void destroyLayer( const Layer& layer )
{
    free( layer.outputMat );
    free( layer.errorMat );
    cudaErrorCheck( cudaFree( layer.dOutputMat ) );
    cudaErrorCheck( cudaFree( layer.dErrorMat ) );
}
