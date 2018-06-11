
#ifndef _ACTIVATION_FUNCTION_HPP_
#define _ACTIVATION_FUNCTION_HPP_

#include "Layer.hpp"
#include "Connection.hpp"

namespace MiniNeuralNetwork
{
    struct ActivationFunction
    {
        virtual void forwardOutput(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) = 0;

        virtual void backPropError(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) = 0;

        virtual void computeOutputLayerError(
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cudaStream_t stream ) = 0;

        virtual void updateWeights(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            const float learningParam,
            const float regularParam,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) = 0;

        virtual float computeCost(
            float* dCostMat,
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) = 0;
    };
}

#endif
