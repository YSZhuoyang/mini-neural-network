
#include "act/Relu.hpp"


__global__ void Relu(
    float* __restrict__ dOutputMat,
    const unsigned int subMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    if (dOutputMat[eleId] < 0.0f)
        dOutputMat[eleId] = 0.0f;
}

__global__ void DRelu(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    float error = (float) (dOutputMat[eleId] >= 0.0f);
    dErrorMat[eleId] *= error;
}


using namespace MiniNeuralNetwork;

unsigned short ReluFunction::standardizeOutput( float output )
{
    throw( "Cannot standardize output given the output layer uses RELU function" );
}

void ReluFunction::forwardActivate(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cudaStream_t stream )
{
}

void ReluFunction::backwardActivate(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cudaStream_t stream )
{
    // DRelu<<<
    //     targetLayer.sigKernalConfig.gridDim,
    //     targetLayer.sigKernalConfig.blockDim,
    //     0,
    //     stream >>>(
    //         targetLayer.dErrorMat,
    //         targetLayer.dOutputMat,
    //         targetLayer.errorMatSize );
    // cudaErrorCheck( cudaGetLastError() );
}

void ReluFunction::computeOutputLayerError(
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cudaStream_t stream )
{
    throw( "Cannot use Relu in output layer.\n" );
}

float ReluFunction::computeCost(
    float* dCostMat,
    const unsigned short* dClassIndexMat,
    const Layer& outputLayer,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    throw( "Cannot use Relu in output layer.\n" );
}
