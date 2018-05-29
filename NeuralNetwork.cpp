
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    cudaErrorCheck( cudaEventDestroy( trainingComplete ) );
    cudaErrorCheck( cudaEventDestroy( testComplete ) );
    cudaErrorCheck( cudaEventDestroy( forwardPropComplete ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventDestroy( backPropCompletes[i] ) );
    free( backPropCompletes );
    backPropCompletes = nullptr;

    delete[] layerArr;
    layerArr = nullptr;
}


void NeuralNetwork::initLayers(
    const unsigned int* architecture,
    const unsigned int numLayers,
    cublasHandle_t cublasHandle )
{
    this->architecture = architecture;
    this->numLayers = numLayers;
    this->cublasHandle = cublasHandle;

    numHiddenLayers = numLayers - 1;
    layerArr = new Layer[numLayers];

    for (unsigned int i = 0; i < numLayers; i++)
    {
        const LayerType layerType = (i == numLayers - 1) ? OUTPUT_LAYER : HIDDEN_LAYER;

        layerArr[i].init(
            architecture[i],
            architecture[i + 1],
            layerType,
            cublasHandle );
    }

    cudaErrorCheck( cudaEventCreateWithFlags(
        &trainingComplete,
        cudaEventDisableTiming ) );
    cudaErrorCheck( cudaEventCreateWithFlags(
        &testComplete,
        cudaEventDisableTiming ) );
    cudaErrorCheck( cudaEventCreateWithFlags(
        &forwardPropComplete,
        cudaEventDisableTiming ) );
    backPropCompletes =
        (cudaEvent_t*) malloc( numHiddenLayers * sizeof( cudaEvent_t ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventCreateWithFlags(
            &backPropCompletes[i],
            cudaEventDisableTiming ) );
}

void NeuralNetwork::train(
    const float* featureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances,
    const unsigned int maxIter,
    const float learningRate,
    const float regularParam,
    const float costThreshold )
{
    // Init device training data
    float* dFeatureMat = nullptr;
    float* dCostMat = nullptr;
    unsigned short* dClassIndexMat = nullptr;
    const unsigned int trainFeatureMatSize = numInstances * architecture[0];
    const unsigned int classIndexMatSize = numInstances * architecture[numLayers];
    cudaErrorCheck( cudaMalloc(
        (void**) &dFeatureMat,
        trainFeatureMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc(
        (void**) &dCostMat,
        classIndexMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc(
        (void**) &dClassIndexMat,
        classIndexMatSize * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMat,
        featureMat,
        trainFeatureMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassIndexMat,
        classIndexMat,
        classIndexMatSize * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    // Init cuda stream resources
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaErrorCheck( cudaStreamCreate( &stream1 ) );
    cudaErrorCheck( cudaStreamCreate( &stream2 ) );

    // Initialize weight buffer in each layer
    cudaErrorCheck( cudaStreamWaitEvent( stream1, testComplete, 0 ) );
    for (unsigned int i = 0; i < numLayers; i++)
    {
        layerArr[i].initWeightData();
        layerArr[i].initOutputBuffers( numInstances );
    }

    // Start gradient descent
    cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
    float learningParam = -learningRate / (float) numInstances;
    unsigned int iter = 0;
    while (iter++ < maxIter)
    {
        forwardProp( dFeatureMat, dClassIndexMat, stream1 );
        backProp( dFeatureMat, learningParam, regularParam, stream1, stream2 );

        printf( "\n" );
    }
    cudaErrorCheck( cudaEventRecord( trainingComplete, stream1 ) );

    // Sum up cost
    float costSum =
        layerArr[numHiddenLayers].computeCost( dCostMat, dClassIndexMat, stream1 );
    // cudaErrorCheck( cudaStreamSynchronize( stream1 ) );
    printf( "Cost: %f\n", costSum );

    // Release cuda stream resources
    cudaErrorCheck( cudaStreamDestroy( stream1 ) );
    cudaErrorCheck( cudaStreamDestroy( stream2 ) );

    // Release training resources
    cudaErrorCheck( cudaFree( dFeatureMat ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    cudaErrorCheck( cudaFree( dCostMat ) );
    dFeatureMat = nullptr;
    dClassIndexMat = nullptr;
    dCostMat = nullptr;
}

void NeuralNetwork::test(
    const float* featureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances )
{
    cudaStream_t stream;
    cudaErrorCheck( cudaStreamCreate( &stream ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream, trainingComplete, 0 ) );
    cublasErrorCheck( cublasSetStream( cublasHandle, stream ) );

    // Prepare buffers in each layer
    for (unsigned int i = 0; i < numLayers; i++)
        layerArr[i].initOutputBuffers( numInstances );

    // Init device test data
    float* dFeatureMat = nullptr;
    unsigned short* dClassIndexMat = nullptr;
    const unsigned int testFeatureMatSize = numInstances * architecture[0];
    const unsigned int classIndexMatSize = numInstances * architecture[numLayers];
    cudaErrorCheck( cudaMalloc(
        (void**) &dFeatureMat,
        testFeatureMatSize * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc(
        (void**) &dClassIndexMat,
        classIndexMatSize * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMat,
        featureMat,
        testFeatureMatSize * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassIndexMat,
        classIndexMat,
        classIndexMatSize * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    // Classify
    forwardProp( dFeatureMat, dClassIndexMat, stream );

    // Compute accuracy
    unsigned int correctCounter = 0;
    unsigned int numOutputFeas = architecture[numLayers];
    float* outputMat = layerArr[numHiddenLayers].getOutputPtr();
    float* dOutputMat = layerArr[numHiddenLayers].getDOutputPtr();

    cudaErrorCheck( cudaMemcpyAsync(
        outputMat,
        dOutputMat,
        classIndexMatSize * sizeof( float ),
        cudaMemcpyDeviceToHost,
        stream ) );
    cudaErrorCheck( cudaStreamSynchronize( stream ) );
    for (unsigned int i = 0; i < numInstances; i++)
    {
        bool correct;
        if (numOutputFeas == 1)
            correct = classIndexMat[i] == (unsigned short) std::lroundf(outputMat[i]);
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
    cudaErrorCheck( cudaStreamDestroy( stream ) );

    // Release test resources
    cudaErrorCheck( cudaFree( dFeatureMat ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    dFeatureMat = nullptr;
    dClassIndexMat = nullptr;
}

inline void NeuralNetwork::forwardProp(
    const float* dFeatureMat,
    const unsigned short* dClassIndexMat,
    cudaStream_t stream )
{
    // Forward propagation
    const float* dInputMat = dFeatureMat;
    for (unsigned int i = 0; i < numLayers; i++)
    {
        printf( "layer %d: forward output ...\n", i );
        dInputMat = layerArr[i].forwardOutput( dInputMat, stream );
    }
    layerArr[numHiddenLayers].computeOutputLayerError(
        dClassIndexMat,
        stream );

    cudaErrorCheck( cudaEventRecord( forwardPropComplete, stream ) );
}

inline void NeuralNetwork::backProp(
    const float* dFeatureMat,
    const float learningParam,
    const float regularParam,
    cudaStream_t stream1,
    cudaStream_t stream2 )
{
    cudaErrorCheck( cudaStreamWaitEvent( stream2, forwardPropComplete, 0 ) );
    // Backword propagation
    for (unsigned int i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer %d: back propagate ...\n", i );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream2 ) );
        layerArr[i - 1].backPropError(
            layerArr[i].getDErrorPtr(),
            layerArr[i].getDWeightPtr(),
            layerArr[i].getNumNodes(),
            stream2 );
        cudaErrorCheck( cudaEventRecord( backPropCompletes[i - 1], stream2 ) );

        printf( "layer %d: update weights ...\n", i );
        cudaErrorCheck( cudaStreamWaitEvent( stream1, backPropCompletes[i - 1], 0 ) );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
        layerArr[i].updateWeights(
            layerArr[i - 1].getDOutputPtr(),
            learningParam,
            regularParam,
            stream1 );
    }

    printf( "layer 0: update weights ...\n" );
    layerArr[0].updateWeights(
        dFeatureMat,
        learningParam,
        regularParam,
        stream1 );
}
