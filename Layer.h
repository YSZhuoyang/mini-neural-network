
#ifndef _LAYER_H_
#define _LAYER_H_

#include "Helper.h"


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    Layer initializeLayer(
        const unsigned int numFeatures,
        const unsigned int numInstances,
        const LayerType layerType );

    void destroyLayer( const Layer& layer );
}

#endif
