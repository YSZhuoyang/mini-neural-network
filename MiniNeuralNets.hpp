
#ifndef MINI_NEURAL_NETS_HPP
#define MINI_NEURAL_NETS_HPP

#include "ActivationFunction.hpp"

namespace MiniNeuralNetwork
{
    struct MiniNeuralNets
    {
        MiniNeuralNets(
            const std::vector<unsigned int>& architecture,
            ActivationFunction* actFunction );
        ~MiniNeuralNets();

        Layer* initializeLayers( const unsigned int numInstances );
        void destroyLayers( Layer* layers );
        void initializeConnections();
        void destroyConnections();

        unsigned int* architecture             = nullptr;
        Connection* connections                = nullptr;
        ActivationFunction* activationFunction = nullptr;
        unsigned short numLayers               = 0;
        unsigned short numHiddenLayers         = 0;
        unsigned short numConnections          = 0;
    };
}

#endif
