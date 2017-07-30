
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

    unsigned int architecture[2];
    architecture[0] = trainSetImporter.GetNumFeatures();
    architecture[1] = 1;
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        1,
        architecture,
        cublasHandle );
    neuralNetwork.train(
        trainSetImporter.GetFeatureMatTrans(),
        trainSetImporter.GetClassIndex(),
        1,
        0.1f,
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
