
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    cudaErrorCheck( cudaStreamDestroy( stream1 ) );
    cudaErrorCheck( cudaStreamDestroy( stream2 ) );

    cudaErrorCheck( cudaEventDestroy( forwardPropComplete ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventDestroy( backPropCompletes[i] ) );
    free( backPropCompletes );
    backPropCompletes = nullptr;

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
    cudaErrorCheck( cudaEventCreateWithFlags( &forwardPropComplete, cudaEventDisableTiming ) );
    backPropCompletes = (cudaEvent_t *) malloc( numHiddenLayers * sizeof( cudaEvent_t ) );
    for (unsigned int i = 0; i < numHiddenLayers; i++)
        cudaErrorCheck( cudaEventCreateWithFlags( &backPropCompletes[i], cudaEventDisableTiming ) );
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
    cublasErrorCheck( cublasSetStream( cublasHandle, stream1 ) );
    while (iter++ < maxIter)
    {
        forwardProp();
        backProp( learningParam );

        printf( "\n" );
    }

    // Copy from device to host
    // For testing gradient descent
    float* outputMat = layerArr[numHiddenLayers].getOutputPtr();
    float* dOutputMat = layerArr[numHiddenLayers].getDOutputPtr();
    unsigned int numFeaturesOut = layerArr[numHiddenLayers].getNumNodes();
    unsigned int outputMatSize = numFeaturesOut * numInstances;
    cudaErrorCheck( cudaMemcpy(
        outputMat,
        dOutputMat,
        outputMatSize * sizeof( float ),
        cudaMemcpyDeviceToHost ) );

    float costSum = 0.0f;
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int j = 0; j < numFeaturesOut; j++)
            costSum -= (classIndexVec[i]) ?
                logf(outputMat[i * numFeaturesOut + j]) : logf(1.0f - outputMat[i * numFeaturesOut + j]);

    printf( "Cost: %f\n", costSum );
}

void NeuralNetwork::forwardProp()
{
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

    cudaErrorCheck( cudaEventRecord( forwardPropComplete, stream1 ) );
    cudaErrorCheck( cudaStreamWaitEvent( stream2, forwardPropComplete, 0 ) );
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
            layerArr[i].getNumNodes(),
            stream2 );
        cudaErrorCheck( cudaEventRecord( backPropCompletes[i - 1], stream2 ) );

        printf( "layer %d: update weights ...\n", i );
        cudaErrorCheck( cudaStreamWaitEvent( stream1, backPropCompletes[i - 1], 0 ) );
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
