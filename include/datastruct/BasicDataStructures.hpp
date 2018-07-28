
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef BASIC_DATA_STRUCTURES_HPP
#define BASIC_DATA_STRUCTURES_HPP


namespace BasicDataStructures
{
    struct Instance
    {
        float* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        float min;
        float max;
        float mean;
    };

    struct KernalConfig
    {
        dim3 blockDim;
        dim3 gridDim;
    };
}

#endif
