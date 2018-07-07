
#include "Sigmoid.hpp"


__global__ void Sigmoid(
    float* __restrict__ dOutputMat,
    const unsigned int subMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float output = dOutputMat[eleId];
    dOutputMat[eleId] = 1.0f / (1.0f + expf( -output ));
}

__global__ void DSigmoid(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float error = dOutputMat[eleId] * (1.0f - dOutputMat[eleId]);
    dErrorMat[eleId] *= error;
}

__global__ void ComputeSigmoidOutputLayerError(
    float* __restrict__ dErrorMat,
    float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    dErrorMat[eleId] = dOutputMat[eleId] - (float) dClassIndexMat[eleId];
}

__global__ void ComputeSigmoidCost(
    float* __restrict__ dCostMat,
    const float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int costMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= costMatSize) return;

    // Note that each element in dCostMat is always > 0
    dCostMat[eleId] = (dClassIndexMat[eleId]) ?
        -logf( dOutputMat[eleId] ) : -logf( 1.0f - dOutputMat[eleId] );
}

using namespace MiniNeuralNetwork;

unsigned short SigmoidFunction::standardizeOutput( float output )
{
    return (unsigned short) std::lroundf( output );
}

void SigmoidFunction::forwardOutput(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Multiply input matrix by weight matrix
    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numInstances,
        connection.numFeaturesOut,
        connection.numFeaturesIn,
        &alpha,
        sourceLayer.dOutputMat,
        numInstances,
        connection.dWeightMat,
        connection.numFeaturesIn,
        &beta,
        targetLayer.dOutputMat,
        numInstances ) );
    Sigmoid<<<
        targetLayer.sigKernalConfig.gridDim,
        targetLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            targetLayer.dOutputMat,
            // Error mat size = output mat size without X0s
            targetLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void SigmoidFunction::backPropError(
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
        targetLayer.numNodes,
        sourceLayer.numNodes,
        &alpha,
        sourceLayer.dErrorMat,
        numInstances,
        connection.dWeightMat,
        targetLayer.numFeatures,
        &beta,
        targetLayer.dErrorMat,
        numInstances ) );
    DSigmoid<<<
        targetLayer.sigKernalConfig.gridDim,
        targetLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            targetLayer.dErrorMat,
            targetLayer.dOutputMat,
            targetLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void SigmoidFunction::computeOutputLayerError(
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cudaStream_t stream )
{
    if (outputLayer.layerType != OUTPUT_LAYER)
        throw( "computeOutputLayerError() can only be called by output layer.\n" );

    ComputeSigmoidOutputLayerError<<<
        outputLayer.ccKernalConfig.gridDim,
        outputLayer.ccKernalConfig.blockDim,
        0,
        stream >>>(
            outputLayer.dErrorMat,
            outputLayer.dOutputMat,
            dClassIndexMat,
            outputLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

float SigmoidFunction::computeCost(
    float* dCostMat,
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    if (outputLayer.layerType != OUTPUT_LAYER)
        throw( "computeCost() can only be called by output layer.\n" );

    float costSum = 0.0f;
    ComputeSigmoidCost<<<
        outputLayer.sigKernalConfig.gridDim,
        outputLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            dCostMat,
            outputLayer.dOutputMat,
            dClassIndexMat,
            outputLayer.outputMatSize );
    cudaErrorCheck( cudaGetLastError() );
    // Sum up absolute values
    cublasErrorCheck( cublasSasum(
        cublasHandle,
        outputLayer.outputMatSize,
        dCostMat,
        1,
        &costSum ) );
    
    return costSum;
}
