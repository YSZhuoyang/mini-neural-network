
#ifndef HYPER_TANGENT_HPP
#define HYPER_TANGENT_HPP

#include "ActivationFunction.hpp"


__global__ void HyperTangent(
    float* __restrict__ dOutputMat,
    const unsigned int subMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float expOfDotProd = expf( dOutputMat[eleId] );
    float expOfMinorsDotProd = 1.0f / expOfDotProd;
    dOutputMat[eleId] = ( expOfDotProd - expOfMinorsDotProd ) / ( expOfDotProd + expOfMinorsDotProd );
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


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct HyperTangentFunction : public ActivationFunction
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
    };
}

#endif
