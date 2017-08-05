
#include "ArffImporter.h"
#include "NeuralNetwork.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Number of layers excluding input layer
    const unsigned int numLayers = 3;
    const unsigned int numClasses = trainSetImporter.GetNumClasses();
    unsigned int architecture[numLayers + 1];
    // Number of features in each layer including input layer
    architecture[0] = trainSetImporter.GetNumFeatures();
    architecture[1] = 501;
    architecture[2] = 51;
    architecture[3] = (numClasses == 2) ? 1 : numClasses;
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        numLayers,
        architecture,
        cublasHandle );

    time_t start, end;
    double dif;
    time( &start );

    neuralNetwork.train(
        trainSetImporter.GetFeatureMatTrans(),
        trainSetImporter.GetClassIndexMat(),
        200,
        0.2f,
        0.1f,
        1.0f );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken: %.2lf seconds.\n", dif );

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
