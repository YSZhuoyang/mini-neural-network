# MiniNeuralNetwork

A general perception neural network written in CUDA.

## Variable Explanation

        Note: Input layer is described as inputFeatureMatrix which is not included below.

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
