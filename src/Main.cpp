
#include "util/ArffImporter.hpp"
#include "act/Sigmoid.hpp"
#include "act/HyperTangent.hpp"
#include "trainer/GradientDescent.hpp"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    ArffImporter testSetImporter;
    testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    using namespace MiniNeuralNetwork;

    // Specify architecture
    const unsigned int numClasses = trainSetImporter.GetNumClasses();
    // Number of features in each layer, including bias input
    std::vector<unsigned int> architecture
    {
        trainSetImporter.GetNumFeatures() + 1,
        7,
        (numClasses == 2) ? 1 : numClasses
    };
    // Determine activation function between layers
    std::shared_ptr<ActivationFunction> sig = std::make_shared<SigmoidFunction>();
    // std::shared_ptr<ActivationFunction> hTan = std::make_shared<HyperTangentFunction>();
    std::vector<std::shared_ptr<ActivationFunction>> activationFunctions
    {
        sig,
        sig
    };

    std::shared_ptr<MiniNeuralNets> miniNeuralNets =
        std::make_shared<MiniNeuralNets>( architecture, activationFunctions );
    Trainer trainer( miniNeuralNets, cublasHandle );

    time_t start, end;
    double dif;
    time( &start );

    trainer.train(
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

    trainer.test(
        testSetImporter.GetFeatureMatTrans(),
        testSetImporter.GetClassIndexMat(),
        testSetImporter.GetNumInstances() );

    // Release CuBLAS context resources
    cublasErrorCheck( cublasDestroy( cublasHandle ) );

    return 0;
}
