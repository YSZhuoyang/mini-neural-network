
#include "include/trainer/GradientDescent.hpp"
// #include "include/cutlass/cutlass/examples/00_basic_gemm.cu"

#include <iostream>
// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "lib/cutlass/cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "lib/cutlass/cutlass/gemm/sgemm_traits.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
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
    cudaStream_t stream )
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
        cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
        cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
        cutlass::Shape<8, 128, 128>            // threadblock tile size
    > GemmTraits;

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
        B,     // matrix B operand
        ldb,
        beta,  // scalar beta
        C,     // source matrix C
        ldc,
        C,     // destination matrix C (may be different memory than source C matrix)
        ldc );

    if (result) {
        std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
        return cudaErrorInvalidValue;
    }

    // Launch the CUTLASS GEMM kernel.
    Gemm::launch(params, stream);

    // Return any errors associated with the launch or cudaSuccess if no error.
    return cudaGetLastError();
}


__global__ void UpdateWeightMat(
    float* __restrict__ dWeightMat,
    const float* __restrict__ dDeltaWeightMat,
    const float learningParam,
    const float regularParam,
    const unsigned int numFeaturesIn,
    const unsigned int weightMatSize )
{
    const unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= weightMatSize) return;

    // Add regularization term excluding bias term
    float regularTerm = ((eleId + 1) % numFeaturesIn == 0) ?
        0.0f : regularParam * dWeightMat[eleId];
    dWeightMat[eleId] += learningParam * (dDeltaWeightMat[eleId] + regularTerm);
}


using namespace MiniNeuralNetwork;

Trainer::Trainer(
    std::shared_ptr<MiniNeuralNets> neuralNets,
    cublasHandle_t cublasHandle )
{
    this->neuralNets = neuralNets;
    this->cublasHandle = cublasHandle;

    cudaErrorCheck( cudaEventCreateWithFlags(
        &trainingComplete,
        cudaEventDisableTiming ) );
    cudaErrorCheck( cudaEventCreateWithFlags(
        &testComplete,
        cudaEventDisableTiming ) );
    cudaErrorCheck( cudaEventCreateWithFlags(
        &forwardPropComplete,
        cudaEventDisableTiming ) );
    // Note: input layer does not have error matrix, thus back propagating
    // error matrix stops at the first hidden layer.
    backPropCompletes =
        (cudaEvent_t*) malloc( neuralNets->numHiddenLayers * sizeof( cudaEvent_t ) );
    for (unsigned short i = 0; i < neuralNets->numHiddenLayers; i++)
        cudaErrorCheck( cudaEventCreateWithFlags(
            &backPropCompletes[i],
            cudaEventDisableTiming ) );
}

Trainer::~Trainer()
{
    cudaErrorCheck( cudaEventDestroy( trainingComplete ) );
    cudaErrorCheck( cudaEventDestroy( testComplete ) );
    cudaErrorCheck( cudaEventDestroy( forwardPropComplete ) );
    for (unsigned short i = 0; i < neuralNets->numHiddenLayers; i++)
        cudaErrorCheck( cudaEventDestroy( backPropCompletes[i] ) );
    free( backPropCompletes );
    backPropCompletes = nullptr;
}

void Trainer::train(
    const float* trainingFeatureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances,
    const unsigned int maxIter,
    const float learningRate,
    const float regularParam,
    const float costThreshold )
{
    const unsigned short numLayers = neuralNets->numLayers;
    const unsigned short numHiddenLayers = neuralNets->numHiddenLayers;

    /******** Init device training data ********/

    // Allocate buffer memory
    neuralNets->initializeConnections();
    Layer* layers = neuralNets->initializeLayers( numInstances );

    const unsigned int classIndexMatSize = numInstances * layers[numLayers - 1].numNodes;
    const unsigned int trainingFeatureMatSize = numInstances * layers[0].numNodes;
    // Init input layer data
    cudaErrorCheck( cudaMemcpyAsync(
        layers[0].dOutputMat,
        trainingFeatureMat,
        trainingFeatureMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    // Init training label data
    unsigned short* dClassIndexMat = nullptr;
    cudaErrorCheck( cudaMalloc(
        (void**) &dClassIndexMat,
        classIndexMatSize * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassIndexMat,
        classIndexMat,
        classIndexMatSize * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    /*******************************************/

    // Init cuda stream resources
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaErrorCheck( cudaStreamCreate( &stream1 ) );
    cudaErrorCheck( cudaStreamCreate( &stream2 ) );

    // Start gradient descent
    cudaErrorCheck( cudaStreamWaitEvent( stream1, testComplete, 0 ) );
    cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
    const float learningParam = -learningRate / (float) numInstances;
    for (unsigned int i = 0; i < maxIter; i++)
    {
        forwardProp( layers, numInstances, stream1 );
        backwardProp(
            layers,
            dClassIndexMat,
            numInstances,
            learningParam,
            regularParam,
            stream1,
            stream2 );

        printf( "\n" );
    }
    cudaErrorCheck( cudaEventRecord( trainingComplete, stream1 ) );

    // Sum up cost
    float* dCostMat = nullptr;
    cudaErrorCheck( cudaMalloc(
        (void**) &dCostMat,
        classIndexMatSize * sizeof( float ) ) );
    std::shared_ptr<ActivationFunction> outputActFunction =
        neuralNets->activationFunctions[numHiddenLayers];
    const float costSum = outputActFunction->computeCost(
        dCostMat,
        dClassIndexMat,
        layers[numLayers - 1],
        cublasHandle,
        stream1 );
    printf( "Cost: %f\n", costSum );

    // Release output memory in each layer
    neuralNets->destroyLayers( layers );

    // Release training resources
    cudaErrorCheck( cudaStreamDestroy( stream1 ) );
    cudaErrorCheck( cudaStreamDestroy( stream2 ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    cudaErrorCheck( cudaFree( dCostMat ) );
    dClassIndexMat = nullptr;
    dCostMat = nullptr;
}

void Trainer::test(
    const float* testFeatureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances )
{
    const unsigned short numLayers = neuralNets->numLayers;
    const unsigned short numHiddenLayers = neuralNets->numHiddenLayers;

    // Init cuda stream resources
    cudaStream_t stream;
    cudaErrorCheck( cudaStreamCreate( &stream ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream, trainingComplete, 0 ) );
    cublasErrorCheck( cublasSetStream( cublasHandle, stream ) );

    /********** Init device test data **********/

    // Allocate buffer memory
    Layer* layers = neuralNets->initializeLayers( numInstances );

    const unsigned int testFeatureMatSize = numInstances * layers[0].numNodes;
    const unsigned int classIndexMatSize = numInstances * layers[numLayers - 1].numNodes;
    // Init input layer data
    cudaErrorCheck( cudaMemcpyAsync(
        layers[0].dOutputMat,
        testFeatureMat,
        testFeatureMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    // Init test label data
    unsigned short* dClassIndexMat = nullptr;
    cudaErrorCheck( cudaMalloc(
        (void**) &dClassIndexMat,
        classIndexMatSize * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassIndexMat,
        classIndexMat,
        classIndexMatSize * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    /*******************************************/

    // Classify
    forwardProp( layers, numInstances, stream );

    // Compute accuracy
    unsigned int correctCounter = 0;
    unsigned int numOutputFeas = layers[numLayers - 1].numNodes;
    float* outputMat = layers[numLayers - 1].outputMat;
    float* dOutputMat = layers[numLayers - 1].dOutputMat;
    std::shared_ptr<ActivationFunction> outputActFunction =
        neuralNets->activationFunctions[numHiddenLayers];

    cudaErrorCheck( cudaMemcpy(
        outputMat,
        dOutputMat,
        classIndexMatSize * sizeof( float ),
        cudaMemcpyDeviceToHost ) );

    for (unsigned int i = 0; i < numInstances; i++)
    {
        bool correct;
        if (numOutputFeas == 1)
            correct = classIndexMat[i] ==
                outputActFunction->standardizeOutput( outputMat[i] );
        else
        {
            float max = outputMat[i];
            unsigned int predictedClassIndex = 0;
            for (unsigned int j = 1; j < numOutputFeas; j++)
            {
                if (max < outputMat[numInstances * j + i])
                {
                    max = outputMat[numInstances * j + i];
                    predictedClassIndex = j;
                }
            }
            correct = classIndexMat[predictedClassIndex];
        }
        correctCounter += correct;
    }

    printf( "Accuracy: %f\n", (float) correctCounter / (float) numInstances );
    cudaErrorCheck( cudaEventRecord( testComplete, stream ) );

    // Release output memory in each layer
    neuralNets->destroyLayers( layers );

    // Release test resources
    cudaErrorCheck( cudaStreamDestroy( stream ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    dClassIndexMat = nullptr;
}

inline void Trainer::forwardProp(
    Layer* layers,
    const unsigned int numInstances,
    cudaStream_t stream1 )
{
    const unsigned short numConnections = neuralNets->numConnections;
    Connection* connections = neuralNets->connections;

    for (unsigned short i = 0; i < numConnections; i++)
    {
        printf( "layer %d to layer %d: Forward output ...\n", i, i + 1 );
        // Compute Z(n): W x A(n - 1)
        // Compute A(n): g(Z(n))
        forwardOutput(
            layers[i],
            layers[i + 1],
            connections[i],
            numInstances,
            neuralNets->activationFunctions[i],
            cublasHandle,
            stream1 );
    }
}

inline void Trainer::backwardProp(
    Layer* layers,
    const unsigned short* dClassIndexMat,
    const unsigned int numInstances,
    const float learningParam,
    const float regularParam,
    cudaStream_t stream1,
    cudaStream_t stream2 )
{
    const unsigned short numLayers = neuralNets->numLayers;
    const unsigned short numHiddenLayers = neuralNets->numHiddenLayers;
    Connection* connections = neuralNets->connections;

    // Compute dZ(n): A(n) - Y
    neuralNets->activationFunctions[numHiddenLayers]->computeOutputLayerError(
        dClassIndexMat,
        layers[numLayers - 1],
        stream1 );
    cudaErrorCheck( cudaEventRecord( forwardPropComplete, stream1 ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream2, forwardPropComplete, 0 ) );

    for (unsigned short i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer %d to layer %d: Backprop error ...\n", i + 1, i );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream2 ) );
        // Compute dZ(n - 1): WT x dZ(n) * g'(z(n))
        backPropError(
            layers[i + 1],
            layers[i],
            connections[i],
            numInstances,
            neuralNets->activationFunctions[i - 1],
            cublasHandle,
            stream2 );
        cudaErrorCheck( cudaEventRecord( backPropCompletes[i - 1], stream2 ) );

        printf( "layer %d to layer %d: Update weights ...\n", i + 1, i );
        cudaErrorCheck( cudaStreamWaitEvent( stream1, backPropCompletes[i - 1], 0 ) );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
        // Compute dW(n): dZ(n) x A(n - 1)
        // update W(n): W(n) - lr * dW(n)
        updateWeights(
            layers[i + 1],
            layers[i],
            connections[i],
            numInstances,
            learningParam,
            regularParam,
            cublasHandle,
            stream1 );
    }

    // Compute dW(1): dZ(1) x X
    // update W(1): W(1) - lr * dW(1)
    printf( "layer 1 to layer 0: Update weights ...\n" );
    updateWeights(
        layers[1],
        layers[0],
        connections[0],
        numInstances,
        learningParam,
        regularParam,
        cublasHandle,
        stream1 );
}

inline void Trainer::forwardOutput(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    const std::shared_ptr<ActivationFunction> actFunction,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Multiply input matrix by weight matrix
    // cublasErrorCheck( cublasSgemm(
    //     cublasHandle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_N,
    //     numInstances,
    //     connection.numFeaturesOut,
    //     connection.numFeaturesIn,
    //     &alpha,
    //     sourceLayer.dOutputMat,
    //     numInstances,
    //     connection.dWeightMat,
    //     connection.numFeaturesIn,
    //     &beta,
    //     targetLayer.dOutputMat,
    //     numInstances ) );
    cudaErrorCheck( CutlassSgemmNN(
        numInstances,
        connection.numFeaturesOut,
        connection.numFeaturesIn,
        alpha,
        sourceLayer.dOutputMat,
        numInstances,
        connection.dWeightMat,
        connection.numFeaturesIn,
        beta,
        targetLayer.dOutputMat,
        numInstances,
        stream ) );
    actFunction->forwardActivate(
        targetLayer,
        stream );
}

inline void Trainer::backPropError(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    const std::shared_ptr<ActivationFunction> actFunction,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        numInstances,
        // Exclude bias
        targetLayer.numNodes,
        sourceLayer.numNodes,
        &alpha,
        sourceLayer.dErrorMat,
        numInstances,
        connection.dWeightMat,
        targetLayer.numFeatures,
        &beta,
        targetLayer.dErrorMat,
        numInstances ) );
    actFunction->backwardActivate(
        targetLayer,
        stream );
}

inline void Trainer::updateWeights(
    const Layer& sourceLayer,
    const Layer& targetLayer,
    const Connection& connection,
    const unsigned int numInstances,
    const float learningParam,
    const float regularParam,
    cublasHandle_t cublasHandle,
    cudaStream_t stream )
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute delta weight matrix
    cublasErrorCheck( cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        connection.numFeaturesIn,
        connection.numFeaturesOut,
        numInstances,
        &alpha,
        targetLayer.dOutputMat,
        numInstances,
        sourceLayer.dErrorMat,
        numInstances,
        &beta,
        connection.dDeltaWeightMat,
        connection.numFeaturesIn ) );
    // Update weight matrix
    UpdateWeightMat<<<
        connection.uwKernalConfig.gridDim,
        connection.uwKernalConfig.blockDim,
        0,
        stream >>>(
            connection.dWeightMat,
            connection.dDeltaWeightMat,
            learningParam,
            regularParam,
            connection.numFeaturesIn,
            connection.weightMatSize );
    cudaErrorCheck( cudaGetLastError() );

    // Copy from device to host
    // For testing gradient descent
    // cudaErrorCheck( cudaMemcpy(
    //     weightMat,
    //     dWeightMat,
    //     weightMatSize * sizeof( float ),
    //     cudaMemcpyDeviceToHost ) );

    // float sum = 0.0f;
    // for (unsigned int i = 0; i < weightMatSize; i++)
    //     sum += weightMat[i];

    // printf( "Back propagate completed, weight sum: %f\n", sum );
}
