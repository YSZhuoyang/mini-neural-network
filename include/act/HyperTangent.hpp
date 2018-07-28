
#ifndef HYPER_TANGENT_HPP
#define HYPER_TANGENT_HPP

#include "ActivationFunction.hpp"


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct HyperTangentFunction : public ActivationFunction
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
