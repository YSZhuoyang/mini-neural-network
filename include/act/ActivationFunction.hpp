
#include "include/model/Layer.hpp"
#include "include/model/Connection.hpp"

#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP


namespace MiniNeuralNetwork
{
    struct ActivationFunction
    {
        // Map output domain to output 0 or 1
        virtual unsigned short standardizeOutput( float output ) = 0;

        virtual void forwardActivate(
            const Layer& targetLayer,
            cudaStream_t stream ) = 0;
        
        virtual void backwardActivate(
            const Layer& targetLayer,
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
    };
}

#endif
