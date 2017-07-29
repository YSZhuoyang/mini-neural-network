
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    delete[] layerArr;
    layerArr = nullptr;

    cudaFree( dFeatureMat );
    cudaFree( dClassIndexVec );
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

    unsigned int iter = 0;
    while (iter++ < maxIter)
    {
        forwardProp();
        // backProp( learningRate );

        printf( "\n" );
    }
}

void NeuralNetwork::forwardProp()
{
    // Forward propagation
    const float* dInputMat = dFeatureMat;
    for (unsigned int i = 0; i < numLayers; i++)
    {
        printf( "layer: %d forward output ...\n", i );
        dInputMat = layerArr[i].forwardOutput( dInputMat );
        // printf( "output: %f\n", inputMat[0] );
    }
    layerArr[numHiddenLayers].computeOutputLayerError( dClassIndexVec, classIndexVec );
}

void NeuralNetwork::backProp(
    const float learningRate )
{
    // Backword propagation
    for (unsigned int i = numHiddenLayers; i > 0; i--)
    {
        printf( "layer: %d back propagate ...\n", i );
        layerArr[i].backPropError(
            layerArr[i - 1].getDErrorPtr(),
            layerArr[i - 1].getDOutputPtr() );
        printf( "layer: %d update weights ...\n", i );
        layerArr[i].updateWeights(
            layerArr[i - 1].getDOutputPtr(),
            learningRate );
        // printf( "Weight: %f\n", layerArr[i].getWeightPtr()[0] );

        // float sum = 0.0f;
        // for (int i = 0; i < 10; i++)
        //     for (int j = 0; j < 1001; j++)
        //         sum += layerArr[i].getWeightPtr()[i * 1001 + j];
        // printf( "Weight sum: %f\n", sum );
    }

    printf( "layer: 0 update weights ...\n" );
    layerArr[0].updateWeights(
        dFeatureMat,
        learningRate );
    // printf( "Weight: %f\n", layerArr[0].getWeightPtr()[0] );

    // float sum = 0.0f;
    // for (int i = 0; i < 1; i++)
    //     for (int j = 0; j < 3; j++)
    //         sum += layerArr[0].getWeightPtr()[i * 3 + j];
    // printf( "Weight sum: %f\n", sum );
}
