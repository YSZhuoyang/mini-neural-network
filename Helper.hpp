
#ifndef _HELPER_HPP_
#define _HELPER_HPP_

#include "BasicDataStructures.hpp"
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <random>


using namespace BasicDataStructures;

namespace MyHelper
{
#define NUM_BLOCK_THREADS 128

    bool IsLetter( const char c );
    bool StrEqualCaseSen( const char* str1, const char* str2 );
    bool StrEqualCaseInsen( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    Instance Tokenize(
        const char* str,
        const std::vector<NumericAttr>& featureVec );

    unsigned int getIndexOfMax(
        const unsigned int* uintArray,
        const unsigned int length );
    // Consume a sorted array, remove duplicates in place, 
    // and return the number of unique elements.
    unsigned int removeDuplicates(
        float* sortedArr,
        unsigned int length );
    void cudaErrorCheck( cudaError_t cudaStatus );
    void cublasErrorCheck( cublasStatus_t cublasStatus );
}

#endif
