
#ifndef _MINI_NEURAL_NETS_HPP_
#define _MINI_NEURAL_NETS_HPP_

#include "Layer.hpp"
#include "Connection.hpp"

namespace MiniNeuralNetwork
{
    struct MiniNeuralNets
    {
        MiniNeuralNets( const std::vector<unsigned int>& architecture );
        ~MiniNeuralNets();

        Layer* initializeLayers( const unsigned int numInstances );
        void destroyLayers( Layer* layers );
        void initializeConnections();
        void destroyConnections();

        unsigned int* architecture       = nullptr;
        Connection* connections          = nullptr;
        unsigned short numLayers         = 0;
        unsigned short numHiddenLayers   = 0;
        unsigned short numConnections    = 0;
    };
}

#endif
