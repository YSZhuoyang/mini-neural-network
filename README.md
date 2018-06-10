# MiniNeuralNetwork

A general perception neural network written in CUDA.

## Variable Explanation

                                 inputLayer   hiddenLayers   outputLayer
                                      |             |             |
                                      |       -------------       |
                                      |       |           |       |

             ------  biasInput -----  *       *   *   *
             |
             |           -----------  *       *   *   *   *
        numFeatures      |            *       *   *   *   *       *
             |        numNodes        *       *   *   *   *       *
             |           |            *       *   *   *   *       *
             |           |            *       *   *   *   *
             ------      -----------  *       *   *   *

                                      |                           |
                                      |                           |
                                      |---------numLayers---------|
