
#include "ActivationFunction.hpp"

#ifndef SIGMOID_HPP
#define SIGMOID_HPP


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct SigmoidFunction : public ActivationFunction
    {
        unsigned short standardizeOutput( float output ) final;

        void forwardActivate(
            const Layer& targetLayer,
            cudaStream_t stream ) final;

        void backwardActivate(
            const Layer& targetLayer,
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
