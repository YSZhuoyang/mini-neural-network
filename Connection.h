
#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include "Helper.h"


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    Connection initializeConnection(
        const unsigned int numFeaturesIn,
        const unsigned int numFeaturesOut );
    void destroyConnection( const Connection& connection );
}

#endif
