
#include "Helper.h"


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    Connection initializeConnection(
        const unsigned int numFeaturesIn,
        const unsigned int numFeaturesOut );
    void destroyConnection( const Connection& connection );
}
