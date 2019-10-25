
#include "ActivationFunction.hpp"

#ifndef HYPER_TANGENT_HPP
#define HYPER_TANGENT_HPP


namespace MiniNeuralNetwork
{
    struct HyperTangentFunction : public ActivationFunction
    {
        unsigned short standardizeOutput( float output ) final;

        void forwardActivate(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
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
