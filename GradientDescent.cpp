
#include "GradientDescent.h"


__global__ void Sigmid(
    float* __restrict__ dOutputMat,//dOutputMatOffset
    const unsigned int subMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float output = dOutputMat[eleId];
    dOutputMat[eleId] = 1.0f / (1.0f + expf(-output));
}

__global__ void BackPropError(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMat,//dOutputMatOffset
    const unsigned int errorMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float error = dOutputMat[eleId] * (1.0f - dOutputMat[eleId]);
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

void forwardOutput(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numInstances,
        connection.numFeaturesOut,// numNodes,
        connection.numFeaturesIn,
        &alpha,
        sourceLayer.dOutputMat,
        numInstances,
        connection.dWeightMat,
        connection.numFeaturesIn,
        &beta,
        targetLayer.dOutputMat,// dOutputMatOffset,
        numInstances ) );
    Sigmid<<<
        targetLayer.sigKernalConfig.gridDim,
        targetLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            targetLayer.dOutputMat,// dOutputMatOffset
            // Error mat size = output mat size without X0s
            targetLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void backPropError(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        numInstances,
        // Exclude bias
        sourceLayer.numNodes,
        targetLayer.numNodes,//numNodesNextLayer,
        &alpha,
        targetLayer.dErrorMat,//dErrorMatNextLayer,
        numInstances,
        connection.dWeightMat,//dWeightMatNextLayer + 1,
        sourceLayer.numFeatures,
        &beta,
        sourceLayer.dErrorMat,
        numInstances ) );
    BackPropError<<<
        sourceLayer.sigKernalConfig.gridDim,
        sourceLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            sourceLayer.dErrorMat,
            sourceLayer.dOutputMat,// + numInstances,
            sourceLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void computeOutputLayerError(
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cudaStream_t stream )
{
    if (outputLayer.layerType != OUTPUT_LAYER)
        throw( "computeOutputLayerError() can only be called by output layer.\n" );

    ComputeOutputLayerError<<<
        outputLayer.ccKernalConfig.gridDim,
        outputLayer.ccKernalConfig.blockDim,
        0,
        stream >>>(
            outputLayer.dErrorMat,
            outputLayer.dOutputMat,
            dClassIndexMat,
            outputLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Sum up cost
    // For testing gradient descent
    // float costSum =
    //     layerArr[numHiddenLayers].computeCost( dClassIndexMat, dCostMat, stream1 );
    // printf( "Cost: %f\n", costSum );
}

void updateWeights(
    const Layer& sourceLayer,
    const Connection& connection,
    const unsigned int numInstances,
    const float learningParam,
    const float regularParam,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute delta weight matrix
    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        connection.numFeaturesIn,// numFeaturesIn,
        connection.numFeaturesOut,
        numInstances,
        &alpha,
        sourceLayer.dOutputMat, //dInputMat,
        numInstances,
        sourceLayer.dErrorMat,
        numInstances,
        &beta,
        connection.dDeltaWeightMat,
        connection.numFeaturesIn ) );
    // Update weight matrix
    UpdateWeightMat<<<
        connection.uwKernalConfig.gridDim,
        connection.uwKernalConfig.blockDim,
        0,
        stream >>>(
            connection.dWeightMat,
            connection.dDeltaWeightMat,
            learningParam,
            regularParam,
            connection.numFeaturesIn,
            connection.weightMatSize );
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
