
#include "ActivationFunction.hpp"


__global__ void UpdateWeightMat(
    float* __restrict__ dWeightMat,
    const float* __restrict__ dDeltaWeightMat,
    const float learningParam,
    const float regularParam,
    const unsigned int numFeaturesIn,
    const unsigned int weightMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= weightMatSize) return;

    // Add regularization term excluding bias term
    float regularTerm = ((eleId + 1) % numFeaturesIn == 0) ?
        0.0f : regularParam * dWeightMat[eleId];
    dWeightMat[eleId] += learningParam * (dDeltaWeightMat[eleId] + regularTerm);
}


using namespace MiniNeuralNetwork;

void ActivationFunction::updateWeights(
    const Layer& sourceLayer,
    const Layer& targetLayer,
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
        connection.numFeaturesIn,
        connection.numFeaturesOut,
        numInstances,
        &alpha,
        targetLayer.dOutputMat,
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
