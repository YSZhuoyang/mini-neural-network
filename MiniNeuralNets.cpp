
#include "MiniNeuralNets.hpp"

using namespace MiniNeuralNetwork;

MiniNeuralNets::MiniNeuralNets(
    const std::vector<unsigned int>& architecture,
    std::shared_ptr<ActivationFunction> actFunction )
{
    activationFunction = actFunction;
    this->architecture = std::make_unique<unsigned int[]>(architecture.size());
    std::copy( architecture.begin(), architecture.end(), this->architecture.get() );

    numLayers = architecture.size();
    numHiddenLayers = numLayers - 2;
    numConnections = numLayers - 1;
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
        connections[i] = Connection::initializeConnection( numFeaturesIn, numFeaturesOut );
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
