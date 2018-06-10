
#include "MiniNeuralNets.h"


MiniNeuralNetwork::MiniNeuralNets::MiniNeuralNets()
{

}

MiniNeuralNetwork::MiniNeuralNets::~MiniNeuralNets()
{
    cudaErrorCheck( cudaEventDestroy( trainingComplete ) );
    cudaErrorCheck( cudaEventDestroy( testComplete ) );
    cudaErrorCheck( cudaEventDestroy( forwardPropComplete ) );
    for (unsigned short i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventDestroy( backPropCompletes[i] ) );
    free( backPropCompletes );
    backPropCompletes = nullptr;

    for (unsigned short i = 0; i < numConnections; i++)
        destroyConnection(connections[i]);

    delete[] layers;
    layers = nullptr;
    delete[] connections;
    connections = nullptr;

    free( architecture );
    architecture = nullptr;
}

void MiniNeuralNetwork::MiniNeuralNets::initialize(
    const std::vector<unsigned int>& architecture,
    cublasHandle_t cublasHandle )
{
    this->cublasHandle = cublasHandle;
    this->architecture = (unsigned int*) malloc( architecture.size() * sizeof( unsigned int ) );
    std::copy( architecture.begin(), architecture.end(), this->architecture );

    numLayers = architecture.size();
    numHiddenLayers = numLayers - 2;
    numConnections = numLayers - 1;

    connections = new Connection[numConnections];
    for (unsigned short i = 0; i < numConnections; i++)
    {
        const unsigned int numFeaturesIn = architecture[i];
        const unsigned int numFeaturesOut = (i == numConnections - 1)
            ? architecture[i + 1]
            : architecture[i + 1] - 1;
        connections[i] = initializeConnection(numFeaturesIn, numFeaturesOut);
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
    // Note: input layer does not have error matrix, thus back propagating
    // error matrix stops at the first hidden layer.
    backPropCompletes =
        (cudaEvent_t*) malloc( numHiddenLayers * sizeof( cudaEvent_t ) );
    for (unsigned short i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventCreateWithFlags(
            &backPropCompletes[i],
            cudaEventDisableTiming ) );
}

void MiniNeuralNetwork::MiniNeuralNets::train(
    const float* trainingFeatureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances,
    const unsigned int maxIter,
    const float learningRate,
    const float regularParam,
    const float costThreshold )
{
    // Allocate output memory in each layer
    layers = new Layer[numLayers];
    for (unsigned short i = 0; i < numLayers; i++)
    {
        const LayerType layerType = (i == 0)
            ? INPUT_LAYER
            : (i == numLayers - 1)
                ? OUTPUT_LAYER
                : HIDDEN_LAYER;
        layers[i] = initializeLayer(architecture[i], numInstances, layerType);
    }

    /******** Init device training data ********/

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
    float learningParam = -learningRate / (float) numInstances;
    unsigned int iter = 0;
    while (iter++ < maxIter)
    {
        forwardProp( numInstances, stream1 );
        backwardProp(
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
    float costSum = computeCost(
        dCostMat,
        dClassIndexMat,
        layers[numLayers - 1],
        cublasHandle,
        stream1 );
    printf( "Cost: %f\n", costSum );

    // Release output memory in each layer
    for (unsigned short i = 0; i < numLayers; i++)
        destroyLayer(layers[i]);

    // Release training resources
    cudaErrorCheck( cudaStreamDestroy( stream1 ) );
    cudaErrorCheck( cudaStreamDestroy( stream2 ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    cudaErrorCheck( cudaFree( dCostMat ) );
    dClassIndexMat = nullptr;
    dCostMat = nullptr;
}

void MiniNeuralNetwork::MiniNeuralNets::test(
    const float* testFeatureMat,
    const unsigned short* classIndexMat,
    const unsigned int numInstances )
{
    // Allocate output memory in each layer
    layers = new Layer[numLayers];
    for (unsigned short i = 0; i < numLayers; i++)
    {
        const LayerType layerType = (i == 0)
            ? INPUT_LAYER
            : (i == numLayers - 1)
                ? OUTPUT_LAYER
                : HIDDEN_LAYER;
        layers[i] = initializeLayer(architecture[i], numInstances, layerType);
    }

    // Init cuda stream resources
    cudaStream_t stream;
    cudaErrorCheck( cudaStreamCreate( &stream ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream, trainingComplete, 0 ) );
    cublasErrorCheck( cublasSetStream( cublasHandle, stream ) );

    /********** Init device test data **********/

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
    forwardProp( numInstances, stream );

    // Compute accuracy
    unsigned int correctCounter = 0;
    unsigned int numOutputFeas = layers[numLayers - 1].numNodes;
    float* outputMat = layers[numLayers - 1].outputMat;
    float* dOutputMat = layers[numLayers - 1].dOutputMat;

    cudaErrorCheck( cudaMemcpy(
        outputMat,
        dOutputMat,
        classIndexMatSize * sizeof( float ),
        cudaMemcpyDeviceToHost ) );

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

    // Release output memory in each layer
    for (unsigned short i = 0; i < numLayers; i++)
        destroyLayer(layers[i]);

    // Release test resources
    cudaErrorCheck( cudaStreamDestroy( stream ) );
    cudaErrorCheck( cudaFree( dClassIndexMat ) );
    dClassIndexMat = nullptr;
}

inline void MiniNeuralNetwork::MiniNeuralNets::forwardProp(
    const unsigned int numInstances,
    cudaStream_t stream1 )
{
    for (unsigned short i = 0; i < numConnections; i++)
    {
        printf( "layer %d to layer %d: Forward output ...\n", i, i + 1 );
        forwardOutput(
            layers[i],
            layers[i + 1],
            connections[i],
            numInstances,
            cublasHandle,
            stream1 );
    }
}

inline void MiniNeuralNetwork::MiniNeuralNets::backwardProp(
    const unsigned short* dClassIndexMat,
    const unsigned int numInstances,
    const float learningParam,
    const float regularParam,
    cudaStream_t stream1,
    cudaStream_t stream2 )
{
    computeOutputLayerError(dClassIndexMat, layers[numLayers - 1], stream1);
    cudaErrorCheck( cudaEventRecord( forwardPropComplete, stream1 ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream2, forwardPropComplete, 0 ) );

    for (unsigned short i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer %d to layer %d: Backprop error ...\n", i + 1, i );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream2 ) );
        backPropError(
            layers[i + 1],
            layers[i],
            connections[i],
            numInstances,
            cublasHandle,
            stream2 );
        cudaErrorCheck( cudaEventRecord( backPropCompletes[i - 1], stream2 ) );

        printf( "layer %d to layer %d: Update weights ...\n", i + 1, i );
        cudaErrorCheck( cudaStreamWaitEvent( stream1, backPropCompletes[i - 1], 0 ) );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
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
