
# MiniNeuralNetwork

A general perception neural network written in CUDA.

## Gradient Descent training algorithm

This Gradient Descent algorithm was implemented based on equations from [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning).

### Supported activation function

* Sigmoid.
* Hyper Tangent.

### Gragh representation

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

    * (Node): each node * has an input, an output and an error associated with, input of a bias node is always 1.
    Connection: each connection between a pair of nodes has a weight and an accumulated deltaWeight associated with.
                (deltaWeight is used for updating weight during the gradient descent)

### Steps
    N: number of gradient descent iterations
    M: represents number of training instances
    Nc: number of connections

    ig: index of a gradient descent iteration
    l: index of a layer
    i: index of an instance
    j[l]: index of a node in a layer l
    
    fnIn[l]: number of input features at layer l (including a bias input)
    fnOut[l]: number of output features at layer l, including bias node (fnOut[l] = fnNode[l + 1] = fnIn[l + 1] - 1)
    fnNode[l]: number of nodes at layer l, excluding bias node (fnNode[l] = fnIn[l] - 1)
    
    w[j1[l], j2[l + 1]]: weight associated with a connection between a pair of nodes (at j1[l] and j2[l + 1])
    dw: deltaWeight associated with a connection between a pair of nodes (at j1[l] and j2[l + 1])
    
    r: regularization param
    lr: learning rate
    
    h(x): predicted result based on the hypothesis function (model)
    y: expected result
    
    
    For ig : 1 to N
        For i : 1 to M
            Run forward propagation to compute output of each layer, excluding bias node.
            Run backword propagation to compute errors of each layer, including bias node.
            Accumulate deltaWeight for each connection:
                dw[j[l], j[l + 1]] := dw[j[l], j[l + 1]] + output[l, j[l]] * error[l + 1, j[l + 1]]
        
        For i : 1 to Nc
            Update weight associated with each connection using computed deltaWeight and regularization parm:
                A bias node is connected:
                    w[j1[l], j2[l + 1]] = w[j1[l], j2[l + 1]] + lr * dw[j1[l], j2[l + 1]] / M
                No bias node is connected:
                    w[j1[l], j2[l + 1]] = w[j1[l], j2[l + 1]] + lr * (dw[j1[l], j2[l + 1]] + r * w[j1[l], j2[l + 1]]) / M

#### Forward propagation
Activate each node with a Sigmoid(Logistic) activation function from left to right:

    w[a, b]: weight associated with the connection between node a at layer l and node b at layer l + 1.
    
    z1(x) = w[0, l] * X0 + w[1, l] * x1 + w[2, l] * x2 + ... + w[n, l] * xn
    Sigmoid output: output[l + 1] = g(z1) = 1 / (1 + pow(e, -z1))
    Hyper Tangent output: output[l + 1] = g(z1) = (pow(e, z1) - pow(e, -z1)) / (pow(e, z1) + pow(e, -z1))

#### Backward propagation
  1. Compute error in the output layer:

    e = h(x) - y
    
  2. Backward propagate error in each layer from right to left:

    e[a, b]: error associated with node a at layer b.
    w[a, b]: weight associated with the connection between node a at layer l and node b at layer l + 1.
    
    z2(x) = w[l, 1] * e[1, l + 1] + w[l, 2] * e[2, l + 1] + ... + w[l, n] * e[n, l + 1]
    e[l] = z2 * output[l] * (1 - output[l])
