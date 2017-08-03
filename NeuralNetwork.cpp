
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    cudaErrorCheck( cudaStreamDestroy( stream1 ) );
    cudaErrorCheck( cudaStreamDestroy( stream2 ) );

    cudaErrorCheck( cudaEventDestroy( forwardPropEvent ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventDestroy( backPropEvents[i] ) );
    free( backPropEvents );
    backPropEvents = nullptr;

    delete[] layerArr;
    layerArr = nullptr;

    cudaErrorCheck( cudaFree( dFeatureMat ) );
    cudaErrorCheck( cudaFree( dClassIndexVec ) );
    dFeatureMat = nullptr;
    dClassIndexVec = nullptr;
}


void NeuralNetwork::initLayers(
    const unsigned int numInstances,
    const unsigned int numLayers,
    const unsigned int* architecture,
    cublasHandle_t cublasHandle )
{
    this->architecture = architecture;
    this->numLayers = numLayers;
    this->numInstances = numInstances;
    this->cublasHandle = cublasHandle;

    numHiddenLayers = numLayers - 1;
    layerArr = new Layer[numLayers];

    for (unsigned int i = 0; i < numLayers; i++)
    {
        unsigned short layerType;
        if (i == numLayers - 1) layerType = OUTPUT_LAYER;
        else layerType = HIDDEN_LAYER;

        layerArr[i].init(
            numInstances,
            architecture[i],
            architecture[i + 1],
            layerType,
            cublasHandle );
    }

    cudaErrorCheck( cudaStreamCreate( &stream1 ) );
    cudaErrorCheck( cudaStreamCreate( &stream2 ) );
    cudaErrorCheck( cudaEventCreate( &forwardPropEvent ) );
    backPropEvents = (cudaEvent_t *) malloc( numHiddenLayers * sizeof( cudaEvent_t ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventCreate( &backPropEvents[i] ) );
}

void NeuralNetwork::train(
    const float* featureMat,
    const unsigned short* classIndexVec,
    const unsigned int maxIter,
    const float learningRate,
    const float costThreshold )
{
    this->classIndexVec = classIndexVec;
    const unsigned int numTrainingFeas = architecture[0];
    // Allocate device memo for training data
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMat, numInstances * numTrainingFeas * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassIndexVec, numInstances * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMat,
        featureMat,
        numInstances * numTrainingFeas * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassIndexVec,
        classIndexVec,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    float learningParam = -learningRate / (float) numInstances;
    unsigned int iter = 0;
    while (iter++ < maxIter)
    {
        forwardProp();
        backProp( learningParam );

        printf( "\n" );
    }
}

void NeuralNetwork::forwardProp()
{
    cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );

    // Forward propagation
    const float* dInputMat = dFeatureMat;
    for (unsigned int i = 0; i < numLayers; i++)
    {
        printf( "layer: %d forward output ...\n", i );
        dInputMat = layerArr[i].forwardOutput( dInputMat, stream1 );
    }
    layerArr[numHiddenLayers].computeOutputLayerError(
        dClassIndexVec,
        classIndexVec,
        stream1 );

    cudaErrorCheck( cudaEventRecord( forwardPropEvent ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream2, forwardPropEvent, 0 ) );
}

void NeuralNetwork::backProp( const float learningParam )
{
    // Backword propagation
    for (unsigned int i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer %d: back propagate ...\n", i );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream2 ) );
        layerArr[i - 1].backPropError(
            layerArr[i].getDErrorPtr(),
            layerArr[i].getDWeightPtr(),
            layerArr[i].getNumFeaturesOut(),
            stream2 );
        cudaErrorCheck( cudaEventRecord( backPropEvents[i - 1] ) );

        printf( "layer %d: update weights ...\n", i );
        cudaErrorCheck( cudaStreamWaitEvent( stream1, backPropEvents[i - 1], 0 ) );
        cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
        layerArr[i].updateWeights(
            layerArr[i - 1].getDOutputPtr(),
            learningParam );
    }

    printf( "layer 0: update weights ...\n" );
    layerArr[0].updateWeights(
        dFeatureMat,
        learningParam );
}
