
#include "Layer.h"



__global__ void Sigmid(
    float* __restrict__ dOutputMatOffset,
    const unsigned int subMatSize )
{
    unsigned int eleId = blockDim.x * blockIdx.x + threadIdx.x;
    if (eleId >= subMatSize) return;

    float output = dOutputMatOffset[eleId];
    dOutputMatOffset[eleId] = 1.0f / (1.0f + expf(-output));
}

__global__ void ComputeOutputLayerError(
    float* __restrict__ dErrorMat,
    float* __restrict__ dOutputMat,
    const unsigned short* __restrict__ dClassIndexVec,
    const unsigned int numInstances )
{
    unsigned int instanceId = blockDim.x * blockIdx.x + threadIdx.x;
    if (instanceId >= numInstances) return;

    float output = dOutputMat[instanceId];
    output = 1.0f / (1.0f + expf(-output));
    // dOutputMat[instanceId] = output;
    dErrorMat[instanceId] = output - (float) dClassIndexVec[instanceId];
}


Layer::Layer()
{

}

Layer::~Layer()
{
    free( weightMat );
    free( outputMat );
    free( errorMat );
    cudaFree( dWeightMat );
    cudaFree( dOutputMat );
    cudaFree( dErrorMat );
    weightMat = nullptr;
    outputMat = nullptr;
    errorMat = nullptr;
    dWeightMat = nullptr;
    dOutputMat = nullptr;
    dErrorMat = nullptr;
}


void Layer::init(
    const unsigned int numInstances,
    const unsigned int numFeaturesIn,
    const unsigned int numFeaturesOut,
    const unsigned short layerType,
    cublasHandle_t cublasHandle )
{
    if (layerType == OUTPUT_LAYER && numFeaturesOut == 2)
    {
        printf( "Number of classes in output layer can only be 1"
            "for 2 classes or greater than 2 for more than 2 classes\n" );
        return;
    }

    this->cublasHandle = cublasHandle;
    this->numInstances = numInstances;
    this->numFeaturesOut = numFeaturesOut;
    this->numFeaturesIn = numFeaturesIn;
    this->layerType = layerType;
    numNodes = (layerType == OUTPUT_LAYER) ?
        numFeaturesOut : numFeaturesOut - 1;

    // Allocate host memo
    weightMat = (float*) malloc( numFeaturesIn * numNodes * sizeof( float ) );
    outputMat = (float*) malloc( numInstances * numFeaturesOut * sizeof( float ) );
    errorMat = (float*) malloc( numInstances * numNodes * sizeof( float ) );

    // Setup bias in non-output layer
    if (layerType == HIDDEN_LAYER)
    {
        outputOffset = 1;
        // Fill the first feature with X0 for bias
        for (unsigned int i = 0; i < numInstances; i++)
            outputMat[i] = 1;
            // outputMat[i * numFeaturesOut] = 1;
    }

    // Inie weight matrix
    for (unsigned int i = 0; i < numNodes; i++)
        for (unsigned int j = 0; j < numFeaturesIn; j++)
            weightMat[i * numFeaturesIn + j] = 1.0f;
                // 0.1f * (float) ((i * numFeaturesIn + j) % 10);

    /* Determine block and grid size of ComputeCost kernel */
    if (numInstances > 128)
    {
        ccBlockDim.x = 128;
        ccGridDim.x = (numInstances + 127) / 128;
    }
    else ccBlockDim.x = numInstances;

    unsigned int errorMatSize = numInstances * numNodes;
    if (errorMatSize > 128)
    {
        sigBlockDim.x = 128;
        sigGridDim.x = (errorMatSize + 127) / 128;
    }
    else sigBlockDim.x = errorMatSize;

    // Allocate device memo
    cudaErrorCheck( cudaMalloc( (void**) &dWeightMat, numFeaturesIn * numNodes * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dOutputMat, numInstances * numFeaturesOut * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dErrorMat, numInstances * numNodes * sizeof( float ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dWeightMat,
        weightMat,
        numFeaturesIn * numNodes * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    // Fill in with bias
    cudaErrorCheck( cudaMemcpyAsync(
        dOutputMat,
        outputMat,
        numInstances * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dErrorMat,
        errorMat,
        numInstances * numNodes * sizeof( float ),
        cudaMemcpyHostToDevice ) );

    dOutputMatOffset = (layerType != HIDDEN_LAYER) ? dOutputMat : dOutputMat + numInstances;
}

float* Layer::forwardOutput( const float* dInputMat )
{
    // use cublasCgemm3m ...


    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf( "test\n" );

    // cublasErrorCheck( cublasSgemm(
    //     cublasHandle,
    //     CUBLAS_OP_N, // cublasOperation_t transa,
    //     CUBLAS_OP_N, // cublasOperation_t transb,
    //     numInstances, numNodes, numFeaturesIn, //int m, int n, int k,
    //     &alpha,        //const float           *alpha,
    //     dInputMat, numInstances, //const float           *A, int lda,
    //     dWeightMat, numFeaturesIn, //const float           *B, int ldb,
    //     &beta,        //const float           *beta,
    //     dOutputMatOffset, //float           *C,
    //     numInstances ) ); //int ldc ) );
    Sigmid<<< sigGridDim, sigBlockDim >>>(
        dOutputMatOffset,
        numInstances * numNodes );
    cudaErrorCheck( cudaGetLastError() );

    return dOutputMat;

    // Include bias in non-output layer
    // for (unsigned int i = 0; i < numInstances; i++)
    //     for (unsigned int idNode = 0; idNode < numNodes; idNode++)
    //     {
    //         float sum = 0.0f;
    //         for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
    //             sum += weightMat[idNode * numFeaturesIn + idIn] *
    //                 inputMat[i * numFeaturesIn + idIn];
    //         sum = 1.0f / (1.0f + expf(-sum));
    //         outputMat[numFeaturesOut * i + idNode + outputOffset] = sum;
    //     }

    // return outputMat;
}

void Layer::backPropError(
    float* preLayerErrorMat,
    const float* inputMat )
{
    unsigned int numNodesPreLayer = numFeaturesIn - 1;
    // Ignore bias input
    unsigned int offset = 1;
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int idIn = 0; idIn < numNodesPreLayer; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int idNode = 0; idNode < numNodes; idNode++)
                sum += weightMat[idNode * numFeaturesIn + idIn + offset] *
                    errorMat[numNodes * i + idNode];
            preLayerErrorMat[numNodesPreLayer * i + idIn] =
                sum * inputMat[numFeaturesIn * i + idIn + offset] *
                (1.0f - inputMat[numFeaturesIn * i + idIn + offset]);
        }

    // float sum = 0.0f;
    // for (int i = 0; i < numNodesPreLayer; i++)
    //     for (int j = 0; j < numInstances; j++)
    //         sum += fabs(preLayerErrorMat[j * numNodes + i]);
    // printf( "Pre Error sum: %f\n", sum );

    // printf( "error in: %f\n", preLayerErrorMat[0] );
}

void Layer::updateWeights(
    const float* inputMat,
    const float learningRate )
{
    for (unsigned int idNode = 0; idNode < numNodes; idNode++)
        for (unsigned int idIn = 0; idIn < numFeaturesIn; idIn++)
        {
            float sum = 0.0f;
            for (unsigned int i = 0; i < numInstances; i++)
                sum += inputMat[numFeaturesIn * i + idIn] *
                    errorMat[numNodes * i + idNode];
            weightMat[numFeaturesIn * idNode + idIn] -=
                learningRate / (float) numInstances * sum;
        }

    // float sum = 0.0f;
    // for (int i = 0; i < numNodes; i++)
    //     for (int j = 0; j < numFeaturesIn; j++)
    //         sum += weightMat[i * numFeaturesIn + j];
    // printf( "Weight sum: %f\n", sum );

    printf( "Back propagate completed, weight: %f\n", weightMat[0] );
}

void Layer::computeOutputLayerError(
    const unsigned short* __restrict__ dClassIndexVec,
    const unsigned short* __restrict__ classIndexVec )
{
    if (layerType != OUTPUT_LAYER)
    {
        printf( "computeOutputLayerError() can only be ran by output layer.\n" );
        return;
    }

    ComputeOutputLayerError<<< ccGridDim, ccBlockDim >>>(
        dErrorMat,
        dOutputMat,
        dClassIndexVec,
        numInstances );

    // 2 classes
    // if (numFeaturesOut == 1)
    //     for (unsigned int i = 0; i < numInstances; i++)
    //         errorMat[i] = outputMat[i] - (float) classIndexVec[i];
    // More than 2 classes
    // else
    // {
    //     memmove( errorMat, outputMat, numInstances * numNodes * sizeof( float ) );
    //     for (unsigned int i = 0; i < numInstances; i++)
    //         errorMat[i * numNodes + classIndexVec[i]] -= 1.0f;
    // }

    // Copy from device to host
    cudaErrorCheck( cudaMemcpyAsync(
        errorMat,
        dErrorMat,
        numInstances * numNodes * sizeof( float ),
        cudaMemcpyDeviceToHost ) );

    float costSum = 0.0f;
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int j = 0; j < numNodes; j++)
            costSum -= (classIndexVec[i]) ?
                logf(outputMat[i * numNodes + j]) : logf(1.0f - outputMat[i * numNodes + j]);
            // costSum += fabs(errorMat[i * numNodes + j]);

    printf( "Cost: %f\n", costSum );
}

float* Layer::getDWeightPtr()
{
    return dWeightMat;
}

float* Layer::getDOutputPtr()
{
    return dOutputMat;
}

float* Layer::getDErrorPtr()
{
    return dErrorMat;
}

float* Layer::getWeightPtr()
{
    return weightMat;
}

float* Layer::getOutputPtr()
{
    return outputMat;
}

float* Layer::getErrorPtr()
{
    return errorMat;
}
