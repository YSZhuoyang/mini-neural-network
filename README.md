
# MiniNeuralNetwork

A general perception neural network written in CUDA.

## Gradient Descent

This Gradient Descent algorithm was implemented based on formulas from [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning).

### Supported activation function

* Sigmoid.
* Hyper Tangent.
* Relu.

### Test Environment

* OS: Ubuntu 18.04
* CUDA: 10.1
* GPU: Nvidia GTX 960M

### Build & Run

CD to project root dir and run:

    make clean
    make

    ./bin/gpu_exec

### Variable Annotations

                                 inputLayer   hiddenLayers   outputLayer
                                      |             |             |
                                      |       -------------       |
                                      |       |           |       |

             ------  biasInput -----  *       *   *   *   *
             |
             |           -----------  *       *   *   *   *
        numFeatures      |            *       *   *   *   *       *
             |        numNodes        *       *   *   *   *
             |           |            *       *   *   *   *
             ------      -----------  *       *   *   *

                                      |                           |
                                      |                           |
                                      |---------numLayers---------|

    * (Node): each node has an input / source, an output / target and an error associated with, a bias input node is always 1.
    Connection: each connection between a pair of nodes has a weight and an accumulated deltaWeight associated with.
                (deltaWeight is used for updating weight during the gradient descent)

### Steps
    Data associated with each layer:
        In: input features to a layer (including bias input as 1)
        Out: output features to a layer (including bias output as 1)
        E: error matrix (derivative of cost function with respect to Z) associated with each layer

    Data associated with each connection:
        W: weight matrix
        WT: transposed weight matrix
        dW: deltaWeight matrix associated with a weight matrix calculated with the derivative of the cost function

    M: represents number of training instances
    N: number of gradient descent iterations
    i: index of a gradient descent iteration
    l: layer index

    r: l2 regularization param
    lr: learning rate

    h(x): predicted result given by the model
    y: actual result

    For i : 1 to N
        Run forward propagation to compute output of each layer, excluding bias node.
        Run backword propagation to compute deltaWeight of each layer, including bias node.
        Update weights.

#### Forward propagation
Activate each layer with an activation function (Sigmoid / Hyper Tangent) from left to right:

    Linear output (with bias included in matrices): Z = W x In
    Relu output: Out = g(Z) = max(0, Z)
    Sigmoid output: Out = g(Z) = 1 / (1 + pow(e, -Z))
    Hyper Tangent output: output = g(Z) = (pow(e, Z) - pow(e, -Z)) / (pow(e, Z) + pow(e, -Z))

#### Backward propagation
  1. Compute error matrix of the output layer:

    E = h(x) - y

  2. Backward propagate error in each layer from right to left:

    Relu error: E = Out >= 0 ? 1 : 0
    Sigmoid error: E[l] = WT x E[l + 1] * output[l] * (1 - output[l])
    Hyper Tangent error: E[l] = WT x E[l + 1] * (1 - output[l] * output[l])

  3. Update weights:

    dW = E[l + 1] x Out[l]
    W = W - lr * (dW + r * W) / M

## Dataset and testing

* Sentiment analysis results of 50000 movie reviews from IMDb (25000 for training, 25000 for testing).
* Each data row contains occurrences of top 10/50/200/1000 frequent words in each review and its sentiment result (Positive / Negative).

### Terms of use

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
    Learning Word Vectors for Sentiment Analysis.
    The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
