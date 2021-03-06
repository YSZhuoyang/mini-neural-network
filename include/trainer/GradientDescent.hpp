
#include "model/MiniNeuralNets.hpp"

#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<MiniNeuralNets> neuralNets,
                cublasHandle_t cublasHandle );
            ~Trainer();

            void train(
                const float* featureMat,
                const unsigned short* classIndexMat,
                const unsigned int numInstances,
                const unsigned int maxIter,
                const float learningRate,
                const float regularParam,
                const float costThreshold );
            void test(
                const float* featureMat,
                const unsigned short* classIndexMat,
                const unsigned int numInstances );
            
        private:
            inline void forwardProp(
                Layer* layers,
                const unsigned int numInstances,
                cudaStream_t stream1 );
            inline void backwardProp(
                Layer* layers,
                const unsigned short* dClassIndexMat,
                const unsigned int numInstances,
                const float learningParam,
                const float regularParam,
                cudaStream_t stream1,
                cudaStream_t stream2 );
            inline void forwardOutput(
                const Layer& sourceLayer,
                const Layer& targetLayer,
                const Connection& connection,
                const unsigned int numInstances,
                const std::shared_ptr<ActivationFunction> actFunction,
                cublasHandle_t cublasHandle,
                cudaStream_t stream );
            inline void backPropError(
                const Layer& sourceLayer,
                const Layer& targetLayer,
                const Connection& connection,
                const unsigned int numInstances,
                const std::shared_ptr<ActivationFunction> actFunction,
                cublasHandle_t cublasHandle,
                cudaStream_t stream );
            inline void updateWeights(
                const Layer& sourceLayer,
                const Layer& targetLayer,
                const Connection& connection,
                const unsigned int numInstances,
                const float learningParam,
                const float regularParam,
                cublasHandle_t cublasHandle,
                cudaStream_t stream );

            std::shared_ptr<MiniNeuralNets> neuralNets = nullptr;
            cudaEvent_t* backPropCompletes             = nullptr;
            cudaEvent_t forwardPropComplete;
            cudaEvent_t trainingComplete;
            cudaEvent_t testComplete;
            cublasHandle_t cublasHandle;
    };
}

#endif
