
#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "ActivationFunction.hpp"



namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct SigmoidFunction : public ActivationFunction
    {
        void forwardOutput(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final;

        void backPropError(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final;

        void computeOutputLayerError(
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cudaStream_t stream ) final;

        float computeCost(
            float* dCostMat,
            const unsigned short* dClassIndexMat,
            const Layer& outputLayer,
            cublasHandle_t cublasHandle,
            cudaStream_t stream ) final;
    };
}

#endif
