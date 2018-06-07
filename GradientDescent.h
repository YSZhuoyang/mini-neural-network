
#include "Helper.h"

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
    const Connection& connection,
    const unsigned int numInstances,
    const float learningParam,
    const float regularParam,
    cublasHandle_t cublasHandle,
    cudaStream_t stream );
