
#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include "Layer.hpp"
#include "Connection.hpp"


namespace MiniNeuralNetwork
{
    struct ActivationFunction
    {
        // Map output domain to output 0 or 1
        virtual unsigned short standardizeOutput( float output ) = 0;

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

        virtual float computeCost(
            float* dCostMat,
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) = 0;

        void updateWeights(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            const float learningParam,
            const float regularParam,
            cublasHandle_t cublasHandle,
            cudaStream_t stream );
    };
}

#endif
