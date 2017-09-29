# MiniNeuralNetwork

A general version of neural network written in CUDA.

## Variable Explanation

        Note: Input layer is described as inputFeatureMatrix, not included in here.

                                      hiddenLayers   outputLayer
                                            |             |
                                      -------------       |
                                      |           |       |
        
             ------  biasInput -----  *   *   *
             |
             |           -----------  *   *   *   *
        numFeatures      |            *   *   *   *       *
             |        numNodes        *   *   *   *       *
             |           |            *   *   *   *       *
             |           |            *   *   *   *
             ------      -----------  *   *   *

                                      |                   |
                                      |                   |
                                      |-----numLayers-----|
