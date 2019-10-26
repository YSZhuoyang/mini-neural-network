
#include "act/Sigmoid.hpp"


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

void SigmoidFunction::forwardActivate(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cudaStream_t stream )
{
    cudaErrorCheck( CutlassSgemmNNWithEpilogue<GemmWithSigmoidEpilogue>(
        numInstances,
        connection.numFeaturesOut,
        connection.numFeaturesIn,
        1.0f,
        sourceLayer.dOutputMat,
        numInstances,
        connection.dWeightMat,
        connection.numFeaturesIn,
        0.0f,
        targetLayer.dOutputMat,
        numInstances,
        targetLayer.dOutputMat,
        numInstances,
        stream ) );
}

void SigmoidFunction::backwardActivate(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    cudaStream_t stream )
{
    cudaErrorCheck(CutlassSgemmNNWithEpilogue<GemmWithDSigmoidEpilogue>(
        numInstances,
        // Exclude bias
        targetLayer.numNodes,
        sourceLayer.numNodes,
        1.0f,
        sourceLayer.dErrorMat,
        numInstances,
        connection.dWeightMat,
        targetLayer.numFeatures,
        1.0f,
        targetLayer.dOutputMat,
        numInstances,
        targetLayer.dErrorMat,
        numInstances,
        stream
    ));
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
