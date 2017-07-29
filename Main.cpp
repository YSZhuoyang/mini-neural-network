
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

    const unsigned int architecture[4] = { 51, 1 };//, 41, 11
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        trainSetImporter.GetNumFeatures(),
        1,//3,
        architecture,
        cublasHandle );
    neuralNetwork.train(
        trainSetImporter.GetFeatureMatTrans(),
        trainSetImporter.GetClassIndex(),
        50,
        0.1f,
        1.0f );
    // neuralNetwork.Classify(
    //     testSetImporter.GetInstances(),
    //     testSetImporter.GetNumInstances() );
    // neuralNetwork.Analyze(
    //     "This is bad",
    //     trainSetImporter.GetFeatures(),
    //     trainSetImporter.GetClassAttr() );

    return 0;
}
