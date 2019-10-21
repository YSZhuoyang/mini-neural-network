
#include "include/act/ActivationFunction.hpp"

// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "lib/cutlass/cutlass/gemm/gemm.h"
// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "lib/cutlass/cutlass/gemm/sgemm_traits.h"
#include "lib/cutlass/cutlass/fragment_multiply_add.h"

#include <iostream>

namespace MiniNeuralNetwork
{
using namespace cutlass::gemm;

template <typename Scalar_, typename FragmentMultiplyAdd_ = FragmentMultiplyAdd<Scalar_, Scalar_>>
class SigmoidEpilogueFunctor
{
public:
    // The scalar.
    typedef Scalar_ Scalar;
    // The adapater.
    typedef FragmentMultiplyAdd_ FragmentMultiplyAdd;
    // The accumulator Type
    typedef typename FragmentMultiplyAdd_::ScalarAccum ScalarAccum;

    struct Params
    {
        // The alpha/beta scaling params.
        Scalar alpha, beta;

        Params(Scalar _alpha = 1.0f, Scalar _beta = 0.0f)
            : alpha(_alpha), beta(_beta) {}

        // Initialize the parameters.
        template <typename GemmDesc_>
        CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const &desc)
        {
            alpha = desc.alpha;
            beta = desc.beta;

            return 0;
        }
    };

    CUTLASS_DEVICE SigmoidEpilogueFunctor(Params const &_params) : params(_params)
    {
    }

    /// Method to determine whether the source accumulator matrix C is ever needed. This method
    /// may always safely return true, though better performance is possible if the source accumulator
    /// matrix is never loaded unnecessarily.
    CUTLASS_DEVICE bool source_required() const
    {
        return !is_zero(params.beta);
    }

    /// Evaluate the functor.
    template <typename FragmentA_, typename FragmentB_>
    CUTLASS_DEVICE void evaluate(FragmentA_ const &accum, FragmentB_ &output)
    {
        FragmentMultiplyAdd mad;
        mad.multiply(params.alpha, accum, output);

        for (int i = 0; i < FragmentB_::kElements; ++i)
            output[i] = 1.0f / (1.0f + expf(-output[i]));
    }

    /// Evaluate the functor.
    template <typename FragmentA_, typename FragmentB_>
    CUTLASS_DEVICE void evaluate(FragmentA_ const &accum, FragmentB_ const &old, FragmentB_ &output)
    {
        FragmentMultiplyAdd mad;
        FragmentB_ tmp;

        mad.multiply(params.beta, old, tmp);
        mad.multiply_add(params.alpha, accum, tmp, output);

        for (int i = 0; i < FragmentB_::kElements; ++i)
            output[i] = 1.0f / (1.0f + expf(-output[i])); //max(FragmentB_::Element(0), output[i]);
    }

    Params params;
};

// Define type definition for single-precision CUTLASS GEMM with column-major
// input matrices and 128x128x8 threadblock tile size.
//
// Note, GemmTraits<> is a generic template defined for various general matrix product
// computations within CUTLASS. It is intended to be maximally flexible, and consequently
// it contains numerous template arguments.
//
// To keep the interface manageable, several helpers are defined for plausible compositions
// including the following example for single-precision GEMM. Typical values are used as
// default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
//
// Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
typedef Gemm<SgemmTraits<
    cutlass::MatrixLayout::kColumnMajor, // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor, // layout of B matrix
    cutlass::Shape<8, 32, 64>,           // threadblock tile size
    SigmoidEpilogueFunctor<float>>>
    GemmWithSigmoidEpilogue;

cudaError_t CutlassSigmoidSgemmNN(
    int M,
    int N,
    int K,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float *C,
    int ldc,
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
    typename GemmWithSigmoidEpilogue::Params params;

    int result = params.initialize(
        M,    // GEMM M dimension
        N,    // GEMM N dimension
        K,    // GEMM K dimension
        1.0f, // scalar alpha
        A,    // matrix A operand
        lda,
        B, // matrix B operand
        ldb,
        0.0f, // scalar beta
        C,    // source matrix C
        ldc,
        C, // destination matrix C (may be different memory than source C matrix)
        ldc);

    if (result)
    {
        std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
        return cudaErrorInvalidValue;
    }

    // Launch the CUTLASS GEMM kernel.
    GemmWithSigmoidEpilogue::launch(params, stream);

    // Return any errors associated with the launch or cudaSuccess if no error.
    return cudaGetLastError();
}
} // namespace MiniNeuralNetwork
