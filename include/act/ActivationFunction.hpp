
#include <iostream>

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/fragment_multiply_add.h"

#include "model/Layer.hpp"
#include "model/Connection.hpp"

#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP


namespace MiniNeuralNetwork
{
    template <typename TGemm>
    cudaError_t CutlassSgemmNNWithEpilogue(
        int M,
        int N,
        int K,
        float alpha,
        float const *A,
        int lda,
        float const *B,
        int ldb,
        float beta,
        float *C,
        int ldc,
        float *D,
        int ldd,
        cudaStream_t stream)
    {
        // Construct and initialize CUTLASS GEMM parameters object.
        //
        // One of CUTLASS's design patterns is to define parameters objects that are constructible
        // in host code and passed to kernels by value. These may include pointers, strides, scalars,
        // and other arguments needed by Gemm and its components.
        //
        // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
        // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
        //
        typename TGemm::Params params;

        int result = params.initialize(
            M,    // GEMM M dimension
            N,    // GEMM N dimension
            K,    // GEMM K dimension
            alpha,// scalar alpha
            A,    // matrix A operand
            lda,
            B,    // matrix B operand
            ldb,
            beta, // scalar beta
            C,    // source matrix C
            ldc,
            D,    // destination matrix C (may be different memory than source C matrix)
            ldd);

        if (result)
        {
            std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
            return cudaErrorInvalidValue;
        }

        // Launch the CUTLASS GEMM kernel.
        TGemm::launch(params, stream);

        // Return any errors associated with the launch or cudaSuccess if no error.
        return cudaGetLastError();
    }

    struct ActivationFunction
    {
        // Map output domain to output 0 or 1
        virtual unsigned short standardizeOutput( float output ) = 0;

        virtual void forwardActivate(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cudaStream_t stream ) = 0;
        
        virtual void backwardActivate(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
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
