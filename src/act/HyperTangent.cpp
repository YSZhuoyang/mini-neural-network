
#include "include/act/HyperTangent.hpp"


__global__ void HyperTangent(
    float* __restrict__ dOutputMat,
    const unsigned int subMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    dOutputMat[eleId] = tanhf( dOutputMat[eleId] );
}

__global__ void DHyperTangent(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float error = 1.0f - dOutputMat[eleId] * dOutputMat[eleId];
    dErrorMat[eleId] *= error;
}

__global__ void ComputeHyperTangentOutputLayerError(
    float* __restrict__ dErrorMat,
    float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    dErrorMat[eleId] = dOutputMat[eleId] - (dClassIndexMat[eleId] ? 1.0f : -1.0f);
}

__global__ void ComputeHyperTangentCost(
    float* __restrict__ dCostMat,
    const float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int costMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= costMatSize) return;

    // Note that each element in dCostMat is always > 0
    float offset = dOutputMat[eleId] / 2.0f + 0.5f;
    dCostMat[eleId] = (dClassIndexMat[eleId]) ?
        -logf( offset ) : -logf( 1.0f - offset );
}


using namespace MiniNeuralNetwork;

unsigned short HyperTangentFunction::standardizeOutput( float output )
{
    return (unsigned short) std::lroundf( output / 2.0f + 0.5f );
}

void HyperTangentFunction::forwardActivate(
    const Layer& targetLayer,
    cudaStream_t stream )
{
    HyperTangent<<<
        targetLayer.sigKernalConfig.gridDim,
        targetLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            targetLayer.dOutputMat,
            // Error mat size = output mat size without X0s
            targetLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void HyperTangentFunction::backwardActivate(
    const Layer& targetLayer,
    cudaStream_t stream )
{
    DHyperTangent<<<
        targetLayer.sigKernalConfig.gridDim,
        targetLayer.sigKernalConfig.blockDim,
        0,
        stream >>>(
            targetLayer.dErrorMat,
            targetLayer.dOutputMat,
            targetLayer.errorMatSize );
    cudaErrorCheck( cudaGetLastError() );
}

void HyperTangentFunction::computeOutputLayerError(
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cudaStream_t stream )
{
    if (outputLayer.layerType != OUTPUT_LAYER)
        throw( "computeOutputLayerError() can only be called by output layer.\n" );

    ComputeHyperTangentOutputLayerError<<<
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

float HyperTangentFunction::computeCost(
    float* dCostMat,
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    if (outputLayer.layerType != OUTPUT_LAYER)
        throw( "computeCost() can only be called by output layer.\n" );

    float costSum = 0.0f;
    ComputeHyperTangentCost<<<
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
