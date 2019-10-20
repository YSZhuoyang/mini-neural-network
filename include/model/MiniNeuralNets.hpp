
#include "include/act/ActivationFunction.hpp"

#ifndef MINI_NEURAL_NETS_HPP
#define MINI_NEURAL_NETS_HPP


namespace MiniNeuralNetwork
{
    struct MiniNeuralNets
    {
        MiniNeuralNets(
            const std::vector<unsigned int>& architecture,
            const std::vector<std::shared_ptr<ActivationFunction>>& activationFunctions );
        ~MiniNeuralNets();

        Layer* initializeLayers( const unsigned int numInstances );
        void destroyLayers( Layer* layers );
        void initializeConnections();
        void destroyConnections();

        std::unique_ptr<unsigned int[]> architecture                               = nullptr;
        std::unique_ptr<std::shared_ptr<ActivationFunction>[]> activationFunctions = nullptr;
        Connection* connections                                                    = nullptr;
        unsigned short numLayers                                                   = 0;
        unsigned short numHiddenLayers                                             = 0;
        unsigned short numConnections                                              = 0;
    };
}

#endif
