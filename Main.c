
#include "ArffImporter.h"
#include "NeuralNetwork.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // printf( "num fea: %d\n", trainSetImporter.GetNumFeatures() );

    const unsigned int architecture[3] = { 1001, 11, 1 };
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        2,
        architecture );
    neuralNetwork.train(
        trainSetImporter.GetFeatureMat(),
        trainSetImporter.GetClassIndex(),
        1,
        5,
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
