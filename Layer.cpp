
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
    dim3 blockDim;
    dim3 gridDim;
    if (errorMatSize > NUM_BLOCK_THREADS)
    {
        blockDim.x = NUM_BLOCK_THREADS;
        gridDim.x = (errorMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    }
    else blockDim.x = errorMatSize;

    KernalConfig kernalConfig;
    kernalConfig.blockDim = blockDim;
    kernalConfig.gridDim = gridDim;

    /* Init host buffer data */
    float* errorMat = (float*) malloc( errorMatSize * sizeof( float ) );
    float* outputMat = (float*) malloc( outputMatSize * sizeof( float ) );

    /* Init device buffer data */
    float* dOutputMat;
    float* dErrorMat;
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, outputMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, errorMatSize * sizeof( float ) ) );

    // Init bias input in non-output layer
    if (layerType != OUTPUT_LAYER)
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
    layer.outputMat = outputMat;
    layer.dOutputMat = dOutputMat;
    layer.errorMat = errorMat;
    layer.dErrorMat = dErrorMat;
    layer.numNodes = numNodes;
    layer.numFeatures = numFeatures;
    layer.outputMatSize = outputMatSize;
    layer.errorMatSize = errorMatSize;
    layer.sigKernalConfig = kernalConfig;
    if (layerType == OUTPUT_LAYER)
        layer.ccKernalConfig = kernalConfig;

    return layer;
}

void destroyLayer( const Layer& layer )
{
    free( layer.outputMat );
    free( layer.errorMat );
    cudaErrorCheck( cudaFree( layer.dOutputMat ) );
    cudaErrorCheck( cudaFree( layer.dErrorMat ) );
}
