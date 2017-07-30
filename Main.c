
#include "ArffImporter.h"
#include "NeuralNetwork.h"


int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first50.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    const unsigned int architecture[4] = { 51, 11, 1 };
    NeuralNetwork neuralNetwork;
    neuralNetwork.initLayers(
        trainSetImporter.GetNumInstances(),
        2,
        architecture );
    neuralNetwork.train(
        trainSetImporter.GetFeatureMat(),
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

    return 0;
}
