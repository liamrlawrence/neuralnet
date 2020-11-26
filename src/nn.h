//======================================================================================================================
// File Name    : nn.h
// Description  : a convolutional neural network framework written in pure C using structs and a doubly-linked list
// Authors      : Liam Lawrence
// Created      : November 26, 2020
// License      : MIT License
// Copyright    : (c) 2020, Liam Lawrence
//======================================================================================================================
#ifndef NEURALNET_NN_H
#define NEURALNET_NN_H

#define NN_NUM_HIDDEN_LAYERS    2
#define NN_NUM_INPUTS           4
#define NN_NUM_OUTPUTS          3
#define NN_LEARNING_RATE        0.1f
#define NN_DEFAULT_BIAS         1.0f
#define NN_EPOCHS               10000
#define NN_TRAINING_SIZE        100
#define NN_TESTING_SIZE         50
#define NN_RECORD               0
#define NN_LOAD_WEIGHTS         0
#define NN_SAVE_WEIGHTS         0
#define NN_SAVE_PATH            "../data/net.dat"
#define NN_VISUALIZER_PATH      "../visualizer/data.csv"


// NEURAL NETWORK STRUCTS //
/// @addtogroup nn_framework
/// @{
typedef struct {
    double weighted_input;      ///< Sum of the weighted inputs
    double activation;          ///< activation_function(weighted_input)
    double bias;
    double *weights;            ///< Array of weights, len(weights) = <previous layer size>
    double dz;                  ///< Error derivative
} neuron_t;

typedef struct layer_t {
    struct layer_t *next;       ///< Pointer to the next layer
    struct layer_t *prev;       ///< Pointer to the previous layer
    int size;                   ///< Number of neurons in the layer
    neuron_t *neurons;          ///< Array of neurons
} layer_t;

typedef struct {
    layer_t input;              ///< Input layer
    int hidden_layer_size;      ///< Number of hidden layers
    layer_t *hidden;            ///< Array of hidden layers
    layer_t output;             ///< Output layer
} neural_network_t;
/// @}


/**
 * @brief
 *      Initialize a neural_network_t.
 * @param[in,out]   net         Reference to a neural net.
 * @param[in]       input_n     Number of input neurons.
 * @param[in]       output_n    Number of output neurons.
 * @param[in]       hl_n        Number of hidden layers.
 * @param[in]       nphl        Array of ints corresponding to the number of neurons per hidden layer.
 * @return
 *      Nothing.
 *
 * **Example Initialization**
 * @code
 *      #define NUM_HIDDEN_LAYERS 2
 *
 *      neural_network_t nn;
 *      const int neurons_per_hidden_layer[NUM_HIDDEN_LAYERS] = {4, 6};
 *      init_nn(&nn, NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN_LAYERS, neurons_per_hidden_layer);
 * @endcode
 * @memberof neural_net
 */
void init_nn(neural_network_t *net, const int input_n, const int output_n, const int hl_n, const int *nphl);

/**
 * @brief
 *      Feed and propagate a set of inputs through a network, updating all of the neurons.
 *
 * @param[in,out]   net         Reference to a neural net.
 * @param[in]       inputs      List of values for the input neurons
 * @return
 *      Nothing.
 * @memberof neural_net
 */
void forward_propagate(neural_network_t *net, const double *inputs);

/**
 * @brief
 *      Calculate the error derivative for all of the neurons in a network.
 *
 * @param[in,out]   net         Reference to a neural net.
 * @param[in]       expected    List of expected values to be compared against the network's output neurons.
 * @return
 *      Nothing.
 * @memberof neural_net
 */
void back_propagate(neural_network_t *net, const double *expected);

/**
 * @brief
 *      Update a network's weights and biases based on the error derivative calculated during backpropagation.
 *
 * @param[in,out]   net         Reference to a neural net.
 * @return
 *      Nothing.
 * @memberof neural_net
 */
void update_weights_biases(neural_network_t *net);

#endif //NEURALNET_NN_H
