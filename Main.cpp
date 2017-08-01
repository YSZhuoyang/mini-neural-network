
#include "ArffImporter.h"
#include "NeuralNetwork.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first50.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Number of layers excluding input layer
    const unsigned int numLayers = 2;
    unsigned int architecture[numLayers + 1];
    // Number of features in each layer including input layer
    architecture[0] = trainSetImporter.GetNumFeatures();
    architecture[1] = 11;
    architecture[2] = 1;
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        numLayers,
        architecture,
        cublasHandle );
    neuralNetwork.train(
        trainSetImporter.GetFeatureMatTrans(),
        trainSetImporter.GetClassIndex(),
        2000,
        0.2f,
        1.0f );
    // neuralNetwork.Classify(
    //     testSetImporter.GetInstances(),
    //     testSetImporter.GetNumInstances() );
    // neuralNetwork.Analyze(
    //     "This is bad",
    //     trainSetImporter.GetFeatures(),
    //     trainSetImporter.GetClassAttr() );

    // Release CuBLAS context resources
    cublasErrorCheck( cublasDestroy( cublasHandle ) );

    return 0;
}
