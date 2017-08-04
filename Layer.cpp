
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

__global__ void ComputeOutputLayerError(
    float* __restrict__ dErrorMat,
    float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexVec,
    const unsigned int errorMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float output = dOutputMat[eleId];
    // For testing
    dOutputMat[eleId] = output;
    dErrorMat[eleId] = output - (float) dClassIndexVec[eleId];
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

__global__ void AddRegularizationTerm(
    float* __restrict__ dDeltaWeightMat,
    const float* __restrict__ dWeightMat,
    const float regularParam,
    const unsigned int numFeaturesIn,
    const unsigned int weightMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    // Exclude bias term
    if (eleId >= weightMatSize || eleId % numFeaturesIn == 0) return;

    dDeltaWeightMat[eleId] += regularParam * dWeightMat[eleId];
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
    const unsigned int numInstances,
    const unsigned int numFeaturesIn,
    const unsigned int numFeaturesOut,
    const unsigned short layerType,
    cublasHandle_t cublasHandle )
{
    if (layerType == OUTPUT_LAYER && numFeaturesOut == 2)
    {
        printf( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );
        return;
    }

    this->cublasHandle = cublasHandle;
    this->numInstances = numInstances;
    this->numFeaturesOut = numFeaturesOut;
    this->numFeaturesIn = numFeaturesIn;
    this->layerType = layerType;
    numNodes = (layerType == OUTPUT_LAYER) ?
        numFeaturesOut : numFeaturesOut - 1;
    weightMatSize = numFeaturesIn * numNodes;
    errorMatSize = numInstances * numNodes;
    outputMatSize = numInstances * numFeaturesOut;
    inputMatSize = numInstances * numFeaturesIn;

    // Allocate host memo
    weightMat = (float*) malloc( weightMatSize * sizeof( float ) );
    outputMat = (float*) malloc( outputMatSize * sizeof( float ) );
    errorMat = (float*) malloc( errorMatSize * sizeof( float ) );

    // Setup bias in non-output layer
    if (layerType == HIDDEN_LAYER)
        // Fill the first feature with X0 for bias
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i] = 1.0f;

    // Randomly init weight matrix
    for (unsigned int i = 0; i < weightMatSize; i++)
        weightMat[i] = ((float) (rand() % 101) - 50.0f) / 50.0f;

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

    if (weightMatSize > NUM_BLOCK_THREADS)
    {
        artBlockDim.x = NUM_BLOCK_THREADS;
        artGridDim.x = (weightMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
    }
    else artBlockDim.x = weightMatSize;

    // Allocate device memo
    cudaErrorCheck( cudaMalloc( (void**) &dWeightMat, weightMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dDeltaWeightMat, weightMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, outputMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, errorMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dWeightMat,
        weightMat,
        weightMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    // Fill in with X0 as bias
    cudaErrorCheck( cudaMemcpyAsync(
        dOutputMat,
        outputMat,
        numInstances * sizeof( float ),
        cudaMemcpyHostToDevice ) );

    dOutputMatOffset = (layerType != HIDDEN_LAYER) ? dOutputMat : dOutputMat + numInstances;
}

float* Layer::forwardOutput(
    const float* dInputMat,
    cudaStream_t stream )
{
    // use cublasCgemm3m ...


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
    const float* dNextLayerErrorMat,
    const float* dNextLayerWeightMat,
    const unsigned int numNextLayerFeasOut,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // use cublasCgemm3m ...


    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        numInstances,
        numNodes,
        numNextLayerFeasOut,
        &alpha,
        dNextLayerErrorMat,
        numInstances,
        // Exclude bias
        dNextLayerWeightMat + 1,
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
    const unsigned short* dClassIndexVec,
    const unsigned short* classIndexVec,
    cudaStream_t stream )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be ran by output layer.\n" );
        return;
    }

    ComputeOutputLayerError<<< ccGridDim, ccBlockDim, 0, stream >>>(
        dErrorMat,
        dOutputMat,
        dClassIndexVec,
        errorMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Copy from device to host
    // For testing gradient descent
    // cudaErrorCheck( cudaMemcpy(
    //     outputMat,
    //     dOutputMat,
    //     outputMatSize * sizeof( float ),
    //     cudaMemcpyDeviceToHost ) );

    // float costSum = 0.0f;
    // for (unsigned int i = 0; i < numInstances; i++)
    //     for (unsigned int j = 0; j < numNodes; j++)
    //         costSum -= (classIndexVec[i]) ?
    //             logf(outputMat[i * numNodes + j]) : logf(1.0f - outputMat[i * numNodes + j]);

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
    // Add regularization term
    AddRegularizationTerm<<< artGridDim, artBlockDim, 0, stream >>>(
        dDeltaWeightMat,
        dWeightMat,
        regularParam,
        numFeaturesIn,
        weightMatSize );
    cudaErrorCheck( cudaGetLastError() );
    // Update weight mat
    cublasErrorCheck( cublasSaxpy(
        cublasHandle,
        weightMatSize,
        &learningParam,
        dDeltaWeightMat,
        1,
        dWeightMat,
        1 ) );

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

unsigned int Layer::getNumFeaturesOut()
{
    return numFeaturesOut;
}
