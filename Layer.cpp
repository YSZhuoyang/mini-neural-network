
#include "Layer.h"



__global__ void Sigmid(
    float* __restrict__ dOutputMatOffset,
    const unsigned int subMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float output = dOutputMatOffset[eleId];
    dOutputMatOffset[eleId] = 1.0f / (1.0f + expf(-output));
}

__global__ void BackPropError(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMatOffset,
    const unsigned int errorMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float error = dOutputMatOffset[eleId] * (1.0f - dOutputMatOffset[eleId]);
    dErrorMat[eleId] *= error;
}

__global__ void ComputeOutputLayerError(
    float* __restrict__ dErrorMat,
    float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int errorMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float output = dOutputMat[eleId];
    // For training
    dErrorMat[eleId] = output - (float) dClassIndexMat[eleId];
}

__global__ void UpdateWeightMat(
    float* __restrict__ dWeightMat,
    const float* __restrict__ dDeltaWeightMat,
    const float learningParam,
    const float regularParam,
    const unsigned int numFeaturesIn,
    const unsigned int weightMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= weightMatSize) return;

    // Add regularization term excluding bias term
    float regularTerm = (eleId % numFeaturesIn == 0) ?
        0.0f : regularParam * dWeightMat[eleId];
    dWeightMat[eleId] += learningParam * (dDeltaWeightMat[eleId] + regularTerm);
}

__global__ void ComputeCost(
    float* __restrict__ dCostMat,
    const float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int costMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= costMatSize) return;

    // Note that each element in dCostMat is always > 0
    dCostMat[eleId] = (dClassIndexMat[eleId]) ?
        -logf(dOutputMat[eleId]) : -logf(1.0f - dOutputMat[eleId]);
}


Layer::Layer()
{

}

Layer::~Layer()
{
    free( weightMat );
    free( outputMat );
    free( errorMat );
    cudaErrorCheck( cudaFree( dWeightMat ) );
    cudaErrorCheck( cudaFree( dDeltaWeightMat ) );
    cudaErrorCheck( cudaFree( dOutputMat ) );
    cudaErrorCheck( cudaFree( dErrorMat ) );
    weightMat = nullptr;
    dDeltaWeightMat = nullptr;
    outputMat = nullptr;
    errorMat = nullptr;
    dWeightMat = nullptr;
    dOutputMat = nullptr;
    dErrorMat = nullptr;
}


void Layer::init(
    const unsigned int numFeaturesIn,
    const unsigned int numFeaturesOut,
    const LayerType layerType,
    cublasHandle_t cublasHandle )
{
    if (layerType == OUTPUT_LAYER && numFeaturesOut == 2)
    {
        printf( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );
        return;
    }

    this->cublasHandle = cublasHandle;
    this->numFeaturesOut = numFeaturesOut;
    this->numFeaturesIn = numFeaturesIn;
    this->layerType = layerType;
    numNodes = (layerType == OUTPUT_LAYER) ?
        numFeaturesOut : numFeaturesOut - 1;
    weightMatSize = numFeaturesIn * numNodes;

    // Allocate host memo
    weightMat = (float*) malloc( weightMatSize * sizeof( float ) );
    /* Determine block and grid size of kernel functions */
    if (weightMatSize > NUM_BLOCK_THREADS)
    {
        uwBlockDim.x = NUM_BLOCK_THREADS;
        uwGridDim.x = (weightMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    }
    else uwBlockDim.x = weightMatSize;

    // Allocate device memo
    cudaErrorCheck( cudaMalloc( (void**) &dWeightMat, weightMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dDeltaWeightMat, weightMatSize * sizeof( float ) ) );
}

void Layer::initWeightData( const float initialWeightRange )
{
    // Randomly init weight matrix
    for (unsigned int i = 0; i < weightMatSize; i++)
        weightMat[i] = ((float) (rand() % 1001) - 500.0f) / 500.0f * initialWeightRange;
    cudaErrorCheck( cudaMemcpyAsync(
        dWeightMat,
        weightMat,
        weightMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
}

void Layer::initOutputBuffers( const unsigned int numInstances )
{
    this->numInstances = numInstances;
    errorMatSize = numInstances * numNodes;
    outputMatSize = numInstances * numFeaturesOut;

    /* Determine block and grid size of kernel functions */
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

    // Release previously allocated memo
    free( outputMat );
    free( errorMat );
    cudaErrorCheck( cudaFree( dOutputMat ) );
    cudaErrorCheck( cudaFree( dErrorMat ) );
    outputMat = nullptr;
    errorMat = nullptr;
    dOutputMat = nullptr;
    dErrorMat = nullptr;

    // Init host data
    errorMat = (float*) malloc( errorMatSize * sizeof( float ) );
    outputMat = (float*) malloc( outputMatSize * sizeof( float ) );
    // Setup bias in non-output layer
    if (layerType == HIDDEN_LAYER)
        // Fill the first feature with X0 for bias
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i] = 1.0f;

    // Init device data
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, outputMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, errorMatSize * sizeof( float ) ) );
    dOutputMatOffset = (layerType != HIDDEN_LAYER) ? dOutputMat : dOutputMat + numInstances;
    // Fill in with X0 as bias
    cudaErrorCheck( cudaMemcpyAsync(
        dOutputMat,
        outputMat,
        numInstances * sizeof( float ),
        cudaMemcpyHostToDevice ) );
}

float* Layer::forwardOutput(
    const float* dInputMat,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numInstances,
        numNodes,
        numFeaturesIn,
        &alpha,
        dInputMat,
        numInstances,
        dWeightMat,
        numFeaturesIn,
        &beta,
        dOutputMatOffset,
        numInstances ) );
    Sigmid<<< sigGridDim, sigBlockDim, 0, stream >>>(
        dOutputMatOffset,
        // Error mat size = output mat size without X0s
        errorMatSize );
    cudaErrorCheck( cudaGetLastError() );

    return dOutputMat;
}

void Layer::backPropError(
    const float* dErrorMatNextLayer,
    const float* dWeightMatNextLayer,
    const unsigned int numNodesNextLayer,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        numInstances,
        numNodes,
        numNodesNextLayer,
        &alpha,
        dErrorMatNextLayer,
        numInstances,
        // Exclude bias
        dWeightMatNextLayer + 1,
        numFeaturesOut,
        &beta,
        dErrorMat,
        numInstances ) );
    BackPropError<<< sigGridDim, sigBlockDim, 0, stream >>>(
        dErrorMat,
        dOutputMat + numInstances,
        errorMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Copy from device to host
    // For testing gradient descent
    // cudaErrorCheck( cudaMemcpy(
    //     errorMat,
    //     dErrorMat,
    //     errorMatSize * sizeof( float ),
    //     cudaMemcpyDeviceToHost ) );

    // float sum = 0.0f;
    // for (unsigned int i = 0; i < errorMatSize; i++)
    //     sum += errorMat[i];

    // printf( "Err pre: %f\n", sum );
}

void Layer::computeOutputLayerError(
    const unsigned short* dClassIndexMat,
    cudaStream_t stream )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be called by output layer.\n" );
        return;
    }

    ComputeOutputLayerError<<< ccGridDim, ccBlockDim, 0, stream >>>(
        dErrorMat,
        dOutputMat,
        dClassIndexMat,
        errorMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Sum up cost
    // For testing gradient descent
    // float costSum =
    //     layerArr[numHiddenLayers].computeCost( dClassIndexMat, dCostMat, stream1 );
    // printf( "Cost: %f\n", costSum );
}

void Layer::updateWeights(
    const float* dInputMat,
    const float learningParam,
    const float regularParam,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute delta weight mat
    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        numFeaturesIn,
        numNodes,
        numInstances,
        &alpha,
        dInputMat,
        numInstances,
        dErrorMat,
        numInstances,
        &beta,
        dDeltaWeightMat,
        numFeaturesIn ) );
    // Update weight mat
    UpdateWeightMat<<< uwGridDim, uwBlockDim, 0, stream >>>(
        dWeightMat,
        dDeltaWeightMat,
        learningParam,
        regularParam,
        numFeaturesIn,
        weightMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Copy from device to host
    // For testing gradient descent
    // cudaErrorCheck( cudaMemcpy(
    //     weightMat,
    //     dWeightMat,
    //     weightMatSize * sizeof( float ),
    //     cudaMemcpyDeviceToHost ) );

    // float sum = 0.0f;
    // for (unsigned int i = 0; i < weightMatSize; i++)
    //     sum += weightMat[i];

    // printf( "Back propagate completed, weight sum: %f\n", sum );
}

float Layer::computeCost(
    float* dCostMat,
    const unsigned short* dClassIndexMat,
    cudaStream_t stream )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeCost() can only be called by output layer.\n" );
        return 0.0f;
    }

    float costSum = 0.0f;
    ComputeCost<<< sigGridDim, sigBlockDim, 0, stream >>>(
        dCostMat,
        dOutputMat,
        dClassIndexMat,
        outputMatSize );
    cudaErrorCheck( cudaGetLastError() );
    // Sum up absolute values
    cublasErrorCheck( cublasSasum(
        cublasHandle,
        outputMatSize,
        dCostMat,
        1,
        &costSum ) );
    
    return costSum;
}

float* Layer::getDWeightPtr()
{
    return dWeightMat;
}

float* Layer::getDOutputPtr()
{
    return dOutputMat;
}

float* Layer::getDErrorPtr()
{
    return dErrorMat;
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

unsigned int Layer::getNumNodes()
{
    return numNodes;
}
