
#include "ArffImporter.hpp"


using namespace std;

ArffImporter::ArffImporter()
{
    
}

ArffImporter::~ArffImporter()
{
    free( featureMat );
    free( featureMatTrans );
    free( classIndexMat );

    for (char* classAttr : classVec) free( classAttr );
    classVec.clear();

    for (NumericAttr& feature : featureVec) free( feature.name );
    featureVec.clear();
}

void ArffImporter::BuildFeatureMatrix()
{
    if (featureMat != nullptr || featureMatTrans != nullptr)
        return;

    // Training data does not contain bias X0, which is added in the
    // input layer of the neural nets
    const unsigned short classIndexMatNumRows = (numClasses == 2) ? 1 : numClasses;
    featureMat =
        (float*) malloc( numInstances * numFeatures * sizeof( float ) );
    featureMatTrans =
        (float*) malloc( numInstances * numFeatures * sizeof( float ) );
    classIndexMat = (unsigned short*) calloc(
        classIndexMatNumRows * numInstances,
        sizeof( unsigned short ) );
    for (unsigned int i = 0; i < numInstances; i++)
    {
        float* offset = featureMat + i * numFeatures;
        memmove(
            offset,
            instanceVec[i].featureAttrArray,
            numFeatures * sizeof( float ) );
        free( instanceVec[i].featureAttrArray );

        if (numClasses > 2)
            classIndexMat[instanceVec[i].classIndex * numInstances + i] = 1;
        else
            classIndexMat[i] = instanceVec[i].classIndex;
    }

    Normalize();
    Transpose();
    instanceVec.clear();
}

void ArffImporter::Normalize()
{
    for (unsigned int i = 0; i < numFeatures; i++)
    {
        // Use either range / standard deviation
        float range = featureVec[i].max - featureVec[i].min;
        if (range == 0.0f) continue;

        for (unsigned int j = 0; j < numInstances; j++)
        {
            unsigned int featureIndex = j * numFeatures + i;
            featureMat[featureIndex] =
                (featureMat[featureIndex] - featureVec[i].mean) / range;
        }
    }
}

void ArffImporter::Transpose()
{
    for (unsigned int i = 0; i < numInstances; i++)
        for (unsigned int j = 0; j < numFeatures; j++)
            featureMatTrans[j * numInstances + i] =
                featureMat[i * numFeatures + j];
}

// Need to check string length boundary
void ArffImporter::Read( const char* fileName )
{
    FILE *fp;

    if ((fp = fopen( fileName, "r+" )) == nullptr)
	{
		printf( "File: %s not found!\n", fileName );
		return;
	}

    // Assuming all data types of all features are float
    // and ignoring feature types
    char firstToken[TOKEN_LENGTH_MAX];
    char buffer[READ_LINE_MAX];

    while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
    {
        // Skip empty lines
        if (buffer[0] == '\n') continue;

        int readSize;
        sscanf( buffer, "%s%n", firstToken, &readSize );

        if (StrEqualCaseInsen( firstToken, KEYWORD_ATTRIBUTE ))
        {
            char* featureName = (char*) malloc( TOKEN_LENGTH_MAX );
            char* featureType = (char*) malloc( TOKEN_LENGTH_MAX );
            
            sscanf( buffer + readSize, "%s %s", featureName, featureType );

            // Read feature names
            if (StrEqualCaseInsen( featureType, KEYWORD_NUMERIC ))
            {
                //printf( "Feature name: %s, length: %d \n", 
                //    featureName, GetStrLength( featureName ) );

                NumericAttr feature;
                feature.name       = featureName;
                feature.min        = 0.0;
                feature.max        = 0.0;
                feature.mean       = 0.0;
                featureVec.push_back( feature );
            }
            // Read class names
            else
            {
                // Parse classes attributes
                char* className = (char*) malloc( TOKEN_LENGTH_MAX );
                featureType++;
                
                while (sscanf( featureType, "%[^,}]%n", className, &readSize ) > 0)
                {
                    printf( "Class name: %s \n", className );

                    classVec.push_back( className );
                    className = (char*) malloc( TOKEN_LENGTH_MAX );

                    featureType += readSize + 1;
                }
            }

            continue;
        }
        // Read feature values
        else if (StrEqualCaseInsen( firstToken, KEYWORD_DATA ))
        {
            numFeatures = featureVec.size();
            numClasses = classVec.size();
            unsigned int featureAttrArraySize = numFeatures * sizeof( float );
            float* featureValueSumArr = (float*) calloc( numFeatures, sizeof( float ) );

            while (fgets( buffer, READ_LINE_MAX, fp ) != nullptr)
            {
                unsigned int index = 0;
                unsigned int featureIndex = 0;
                float value;
                
                Instance instance;
                instance.featureAttrArray = (float*) malloc( featureAttrArraySize );

                // Get feature attribute value
                while (sscanf( buffer + index, "%f%n", &value, &readSize ) > 0)
                {
                    if (featureVec[featureIndex].min > value)
                        featureVec[featureIndex].min = value;
                    
                    if (featureVec[featureIndex].max < value)
                        featureVec[featureIndex].max = value;

                    featureValueSumArr[featureIndex] += value;
                    instance.featureAttrArray[featureIndex++] = value;
                    index += readSize + 1;
                }

                // Get class attribute value
                char classValue[TOKEN_LENGTH_MAX];
                sscanf( buffer + index, "%s%n", classValue, &readSize );

                for (unsigned short i = 0; i < numClasses; i++)
                {
                    if (StrEqualCaseSen( classVec[i], classValue ))
                    {
                        instance.classIndex = i;
                        break;
                    }
                }

                instanceVec.push_back( instance );
            }

            unsigned int instanceSize = instanceVec.size();

            // Compute bucket size and mean value for each numerical attribute
            for (unsigned int i = 0; i < numFeatures; i++)
            {
                featureVec[i].mean = featureValueSumArr[i] / (float) instanceSize;

                // printf(
                //     "feature %u, max: %f, min: %f, mean: %f\n",
                //     i,
                //     featureVec[i].max,
                //     featureVec[i].min,
                //     featureVec[i].mean );
            }

            free( featureValueSumArr );
            featureValueSumArr = nullptr;

            break;
        }
    }

    numInstances = instanceVec.size();

    fclose( fp );
    BuildFeatureMatrix();
}

std::vector<char*> ArffImporter::GetClassMeta()
{
    return classVec;
}

std::vector<NumericAttr> ArffImporter::GetFeatureMeta()
{
    return featureVec;
}

float* ArffImporter::GetFeatureMat()
{
    return featureMat;
}

float* ArffImporter::GetFeatureMatTrans()
{
    return featureMatTrans;
}

unsigned short* ArffImporter::GetClassIndexMat()
{
    return classIndexMat;
}

unsigned int ArffImporter::GetNumInstances()
{
    return numInstances;
}

unsigned int ArffImporter::GetNumFeatures()
{
    return numFeatures;
}

unsigned short ArffImporter::GetNumClasses()
{
    return numClasses;
}
