
#include "ActivationFunction.hpp"

#ifndef SIGMOID_HPP
#define SIGMOID_HPP


namespace MiniNeuralNetwork
{
    using namespace cutlass::gemm;

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
    typedef cutlass::gemm::Gemm<SgemmTraits<
        cutlass::MatrixLayout::kColumnMajor, // layout of A matrix
        cutlass::MatrixLayout::kColumnMajor, // layout of B matrix
        cutlass::Shape<8, 32, 64>,           // threadblock tile size
        SigmoidEpilogueFunctor<float>>>
        GemmWithSigmoidEpilogue;

    template <typename Scalar_, typename FragmentMultiplyAdd_ = cutlass::gemm::FragmentMultiplyAdd<Scalar_, Scalar_>>
    class DSigmoidEpilogueFunctor
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

            Params(Scalar _alpha = 1.0f, Scalar _beta = 1.0f)
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

        CUTLASS_DEVICE DSigmoidEpilogueFunctor(Params const &_params) : params(_params)
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
                output[i] = output[i] * (1.0f - output[i]);
        }

        /// Evaluate the functor.
        template <typename FragmentA_, typename FragmentB_>
        CUTLASS_DEVICE void evaluate(FragmentA_ const &accum, FragmentB_ const &old, FragmentB_ &output)
        {
            FragmentMultiplyAdd mad;
            FragmentB_ tmp;

            for (int i = 0; i < FragmentB_::kElements; ++i)
                tmp[i] = old[i] * (1.0f - old[i]);

            // mad.multiply(params.beta, old, tmp);
            mad.multiply(params.alpha, accum, output);
            // mad.multiply_add(params.alpha, accum, tmp, output);

            for (int i = 0; i < FragmentB_::kElements; ++i)
                output[i] *= tmp[i];
        }

        Params params;
    };

    typedef cutlass::gemm::Gemm<SgemmTraits<
        cutlass::MatrixLayout::kColumnMajor, // layout of A matrix
        cutlass::MatrixLayout::kRowMajor,    // layout of B matrix
        cutlass::Shape<8, 32, 64>,           // threadblock tile size
        DSigmoidEpilogueFunctor<float>>>
        GemmWithDSigmoidEpilogue;


    struct SigmoidFunction : public ActivationFunction
    {
        unsigned short standardizeOutput( float output ) final;

        void forwardActivate(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
            cudaStream_t stream ) final;

        void backwardActivate(
            const Layer& sourceLayer,
            const Layer& targetLayer,
            const Connection& connection,
            const unsigned int numInstances,
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
