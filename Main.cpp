
#include "ArffImporter.h"
// #include "NeuralNetwork.h"
#include "MiniNeuralNets.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Specify architecture
    const unsigned int numClasses = trainSetImporter.GetNumClasses();
    // Number of features in each layer including input layer
    std::vector<unsigned int> architecture
    {
        trainSetImporter.GetNumFeatures() + 1,
        7,
        (numClasses == 2) ? 1 : numClasses
    };

    MiniNeuralNets miniNeuralNets;
    miniNeuralNets.initialize(
        architecture,
        cublasHandle );

    time_t start, end;
    double dif;
    time( &start );

    miniNeuralNets.train(
        trainSetImporter.GetFeatureMatTrans(),
        trainSetImporter.GetClassIndexMat(),
        trainSetImporter.GetNumInstances(),
        8000,
        0.2f,
        0.0f,
        1.0f );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken: %.2lf seconds.\n", dif );

    miniNeuralNets.test(
        testSetImporter.GetFeatureMatTrans(),
        testSetImporter.GetClassIndexMat(),
        testSetImporter.GetNumInstances() );

    // Release CuBLAS context resources
    cublasErrorCheck( cublasDestroy( cublasHandle ) );

    return 0;
}
