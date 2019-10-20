
#include "lib/cutlass/cutlass/gemm/gemm.h"
#include "lib/cutlass/cutlass/fragment_multiply_add.h"

namespace MiniNeuralNetwork
{
template <typename Scalar_, typename FragmentMultiplyAdd_ = cutlass::gemm::FragmentMultiplyAdd<Scalar_, Scalar_>>
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
        /// The alpha/beta scaling params.
        Scalar alpha, beta;

        /// Initialize the parameters.
        template <typename GemmDesc_>
        __host__ __device__ int initialize(GemmDesc_ const &desc)
        {
            alpha = desc.alpha;
            beta = desc.beta;

            return 0;
        }
    };

    __device__ SigmoidEpilogueFunctor(Params const &params) : alpha_(params.alpha),
                                                              beta_(params.beta)
    {
    }

    /// Method to determine whether the source accumulator matrix C is ever needed. This method
    /// may always safely return true, though better performance is possible if the source accumulator
    /// matrix is never loaded unnecessarily.
    CUTLASS_DEVICE
    bool source_required() const
    {
        // return !is_zero(params.beta);
        return true;
    }

    /// Evaluate the functor.
    template <typename FragmentA_, typename FragmentB_>
    __device__ void evaluate(FragmentA_ const &accum, FragmentB_ &output)
    {
        FragmentMultiplyAdd mad;
        mad.multiply(alpha_, accum, output);

        for (int i = 0; i < FragmentB_::kElements; ++i)
            output[i] = 1.0f / (1.0f + expf( -output[i] ));
    }

    /// Evaluate the functor.
    template <typename FragmentA_, typename FragmentB_>
    __device__ void evaluate(FragmentA_ const &accum, FragmentB_ const &old, FragmentB_ &output)
    {
        FragmentMultiplyAdd mad;
        FragmentB_ tmp;

        mad.multiply(beta_, old, tmp);
        mad.multiply_add(alpha_, accum, tmp, output);

        for (int i = 0; i < FragmentB_::kElements; ++i)
            output[i] = 1.0f / (1.0f + expf( -output[i] ));//max(FragmentB_::Element(0), output[i]);
    }

private:
    Scalar alpha_, beta_;
};

// typedef cutlass::gemm::Gemm<cutlass::gemm::SgemmTraits<
//     cutlass::MatrixLayout::kRowMajor,
//     cutlass::MatrixLayout::kColumnMajor,
//     cutlass::Shape<8, 32, 64>,
//     SigmoidEpilogueFunctor_epilogue_functor<float>>>
//     SigmoidGemm;

cudaError_t CutlassSigmoidSgemmNN(
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
    cudaStream_t stream)
{
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
    typedef cutlass::gemm::SgemmTraits<
        cutlass::MatrixLayout::kColumnMajor, // layout of A matrix
        cutlass::MatrixLayout::kColumnMajor, // layout of B matrix
        cutlass::Shape<8, 32, 64>,           // threadblock tile size
        SigmoidEpilogueFunctor<float>>
        GemmTraits;

    // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

    // Construct and initialize CUTLASS GEMM parameters object.
    //
    // One of CUTLASS's design patterns is to define parameters objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    typename Gemm::Params params;

    int result = params.initialize(
        M,     // GEMM M dimension
        N,     // GEMM N dimension
        K,     // GEMM K dimension
        alpha, // scalar alpha
        A,     // matrix A operand
        lda,
        B, // matrix B operand
        ldb,
        beta, // scalar beta
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
    Gemm::launch(params, stream);

    // Return any errors associated with the launch or cudaSuccess if no error.
    return cudaGetLastError();
}
} // namespace MiniNeuralNetwork
