
#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "ActivationFunction.hpp"


__global__ void Sigmid(
    float* __restrict__ dOutputMat,
    const unsigned int subMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float output = dOutputMat[eleId];
    dOutputMat[eleId] = 1.0f / (1.0f + expf(-output));
}

__global__ void BackPropError(
    float* __restrict__ dErrorMat,
    const float* __restrict__ dOutputMat,
    const unsigned int errorMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
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
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= errorMatSize) return;

    dErrorMat[eleId] = dOutputMat[eleId] - (float) dClassIndexMat[eleId];
}

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

__global__ void ComputeCost(
    float* __restrict__ dCostMat,
    const float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexMat,
    const unsigned int costMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= costMatSize) return;

    // Note that each element in dCostMat is always > 0
    dCostMat[eleId] = (dClassIndexMat[eleId]) ?
        -logf(dOutputMat[eleId]) : -logf(1.0f - dOutputMat[eleId]);
}


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct SigmoidFunction : public ActivationFunction
    {
        inline void forwardOutput(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;

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
            Sigmid<<<
                targetLayer.sigKernalConfig.gridDim,
                targetLayer.sigKernalConfig.blockDim,
                0,
                stream >>>(
                    targetLayer.dOutputMat,
                    // Error mat size = output mat size without X0s
                    targetLayer.errorMatSize );
            cudaErrorCheck( cudaGetLastError() );
        }

        inline void backPropError(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final
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
            BackPropError<<<
                targetLayer.sigKernalConfig.gridDim,
                targetLayer.sigKernalConfig.blockDim,
                0,
                stream >>>(
                    targetLayer.dErrorMat,
                    targetLayer.dOutputMat,
                    targetLayer.errorMatSize );
            cudaErrorCheck( cudaGetLastError() );
        }

        inline void computeOutputLayerError(
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cudaStream_t stream ) final
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
        }

        inline void updateWeights(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            const float learningParam,
            const float regularParam,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final
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

        inline float computeCost(
            float* dCostMat,
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final
        {
            if (outputLayer.layerType != OUTPUT_LAYER)
                throw( "computeCost() can only be called by output layer.\n" );

            float costSum = 0.0f;
            ComputeCost<<<
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
    };
}

#endif
