
#ifndef CONNECTION_HPP
#define CONNECTION_HPP

#include "util/Helper.hpp"


namespace MiniNeuralNetwork
{
    using namespace MyHelper;

    struct Connection
    {
        static Connection initializeConnection(
            const unsigned int numFeaturesIn,
            const unsigned int numFeaturesOut )
        {
            unsigned int weightMatSize = numFeaturesIn * numFeaturesOut;
            /* Init host buffer data */
            float* weightMat = (float*) malloc( weightMatSize * sizeof( float ) );

            /* Init device buffer data */
            float* dWeightMat;
            float* dDeltaWeightMat;
            cudaErrorCheck( cudaMalloc( (void**) &dWeightMat, weightMatSize * sizeof( float ) ) );
            cudaErrorCheck( cudaMalloc( (void**) &dDeltaWeightMat, weightMatSize * sizeof( float ) ) );

            // Randomly init weight matrix in Gaussian distribution (He et al. 2015)
            std::random_device random;
            std::mt19937 generator( random() );
            std::normal_distribution<float> normalDist;

            const float scalar = sqrtf( 2.0f / (float) numFeaturesIn );
            for (unsigned int i = 0; i < weightMatSize; i++)
                weightMat[i] = ((i + 1) % numFeaturesIn == 0) ? 0.0f : normalDist( generator ) * scalar;

            cudaErrorCheck( cudaMemcpyAsync(
                dWeightMat,
                weightMat,
                weightMatSize * sizeof( float ),
                cudaMemcpyHostToDevice ) );
            
            /* Compute block and grid size of kernel functions */
            dim3 uwBlockDim;
            dim3 uwGridDim;
            if (weightMatSize > NUM_BLOCK_THREADS)
            {
                uwBlockDim.x = NUM_BLOCK_THREADS;
                uwGridDim.x = (weightMatSize + NUM_BLOCK_THREADS - 1) / NUM_BLOCK_THREADS;
            }
            else uwBlockDim.x = weightMatSize;

            KernalConfig uwKernalConfig;
            uwKernalConfig.blockDim = uwBlockDim;
            uwKernalConfig.gridDim = uwGridDim;

            Connection connection;
            connection.weightMatSize = weightMatSize;
            connection.numFeaturesIn = numFeaturesIn;
            connection.numFeaturesOut = numFeaturesOut;
            connection.weightMat = weightMat;
            connection.dWeightMat = dWeightMat;
            connection.dDeltaWeightMat = dDeltaWeightMat;
            connection.uwKernalConfig = uwKernalConfig;

            return connection;
        };

        static void destroyConnection( const Connection& connection )
        {
            free(connection.weightMat);
            cudaErrorCheck( cudaFree( connection.dWeightMat ) );
            cudaErrorCheck( cudaFree( connection.dDeltaWeightMat ) );
        };

        KernalConfig uwKernalConfig;
        float* weightMat            = nullptr;
        float* dWeightMat           = nullptr;
        float* dDeltaWeightMat      = nullptr;
        unsigned int weightMatSize  = 0;
        unsigned int numFeaturesIn  = 0;
        // Number of output features, which is equal to the number of nodes in the next layer
        unsigned int numFeaturesOut = 0;
    };
}

#endif
