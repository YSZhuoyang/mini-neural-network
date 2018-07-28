
#include "include/datastruct/MiniNeuralNets.hpp"


using namespace MiniNeuralNetwork;

MiniNeuralNets::MiniNeuralNets(
    const std::vector<unsigned int>& architecture,
    const std::vector<std::shared_ptr<ActivationFunction>>& activationFunctions )
{
    numConnections = activationFunctions.size();
    numLayers = architecture.size();
    numHiddenLayers = numLayers - 2;

    if (numConnections != numLayers - 1)
        throw("Number of connections does not equal to the number of activation functions");

    this->architecture = std::make_unique<unsigned int[]>( numLayers );
    std::copy(
        architecture.begin(),
        architecture.end(),
        this->architecture.get() );

    this->activationFunctions =
        std::make_unique<std::shared_ptr<ActivationFunction>[]>( numConnections );
    std::copy(
        activationFunctions.begin(),
        activationFunctions.end(),
        this->activationFunctions.get() );
}

MiniNeuralNets::~MiniNeuralNets()
{
    destroyConnections();
}

Layer* MiniNeuralNets::initializeLayers( const unsigned int numInstances )
{
    Layer* layers = new Layer[numLayers];
    for (unsigned short i = 0; i < numLayers; i++)
    {
        const LayerType layerType = (i == 0)
            ? INPUT_LAYER
            : (i == numLayers - 1)
                ? OUTPUT_LAYER
                : HIDDEN_LAYER;
        layers[i] = Layer::initializeLayer( architecture[i], numInstances, layerType );
    }

    return layers;
}

void MiniNeuralNets::destroyLayers( Layer* layers )
{
    if (layers == nullptr) return;

    for (unsigned short i = 0; i < numLayers; i++)
        Layer::destroyLayer( layers[i] );
    delete[] layers;
    layers = nullptr;
}

void MiniNeuralNets::initializeConnections()
{
    destroyConnections();

    connections = new Connection[numConnections];
    for (unsigned short i = 0; i < numConnections; i++)
    {
        const unsigned int numFeaturesIn = architecture[i];
        const unsigned int numFeaturesOut = (i == numConnections - 1)
            ? architecture[i + 1]
            : architecture[i + 1] - 1;
        connections[i] = Connection::initializeConnection(
            numFeaturesIn,
            numFeaturesOut );
    }
}

void MiniNeuralNets::destroyConnections()
{
    if (connections == nullptr) return;

    for (unsigned short i = 0; i < numConnections; i++)
        Connection::destroyConnection( connections[i] );
    delete[] connections;
    connections = nullptr;
}
