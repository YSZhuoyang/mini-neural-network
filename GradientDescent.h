
#include "Helper.h"

namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    void forwardOutput(
        const Layer& sourceLayer,
        const Layer& targetLayer,
        const Connection& connection,
        const unsigned int numInstances,
        cublasHandle_t cublasHandle,
        cudaStream_t stream );

    void backPropError(
        const Layer& sourceLayer,
        const Layer& targetLayer,
        const Connection& connection,
        const unsigned int numInstances,
        cublasHandle_t cublasHandle,
        cudaStream_t stream );

    void computeOutputLayerError(
        const unsigned short* dClassIndexMat,
        const Layer& outputLayer,
        cudaStream_t stream );

    void updateWeights(
        const Layer& sourceLayer,
        const Layer& targetLayer,
        const Connection& connection,
        const unsigned int numInstances,
        const float learningParam,
        const float regularParam,
        cublasHandle_t cublasHandle,
        cudaStream_t stream );

    float computeCost(
        float* dCostMat,
        const unsigned short* dClassIndexMat,
        const Layer& outputLayer,
        cublasHandle_t cublasHandle,
        cudaStream_t stream );
}
