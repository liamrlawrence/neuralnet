//======================================================================================================================
// File Name    : nn.c
// Description  : Implementation of nn.h
// Authors      : Liam Lawrence
// Created      : November 26, 2020
// License      : MIT License
// Copyright    : (c) 2020, Liam Lawrence
//======================================================================================================================
#include "nn.h"
#include <math.h>
#include <stdlib.h>
#include <malloc.h>


// FUNCTION PROTOTYPES //
static inline double rand_double(void);
static void record(neural_network_t *net, int flag);
static double activation_fun(double);
static double activation_prime_fun(double);
static double cost_fun(double predicted, double expected);
static double cost_prime_fun(double predicted, double expected);


// NEURAL NET FUNCTIONS //
void init_nn(neural_network_t *net, const int input_n, const int output_n, const int hl_n, const int *nphl)
{
    int weights_size;

    // Allocate memory for the hidden layers
    net->hidden_layer_size = hl_n;
    net->hidden = malloc(sizeof(layer_t) * net->hidden_layer_size);

    // Initialize the Input layer
    net->input.next = &net->hidden[0];
    net->input.prev = NULL;
    net->input.size = input_n;
    net->input.neurons = malloc(sizeof(neuron_t) * net->input.size);
    for (int j = 0; j < net->input.size; j++) {
        net->input.neurons[j].bias = 0;                                 // unused
        net->input.neurons[j].weights = NULL;                           // unused
    }

    // Initialize the Hidden layers
    for (int l = 0; l < net->hidden_layer_size; l++) {
        net->hidden[l].next = (l == net->hidden_layer_size-1) ? &net->output : &net->hidden[l+1];
        net->hidden[l].prev = (l == 0) ? &net->input : &net->hidden[l-1];
        net->hidden[l].size = nphl[l];
        net->hidden[l].neurons = malloc(sizeof(neuron_t) * net->hidden[l].size);
        weights_size = net->hidden[l].prev->size;
        for (int j = 0; j < net->hidden[l].size; j++) {
            net->hidden[l].neurons[j].bias = NN_DEFAULT_BIAS;
            net->hidden[l].neurons[j].weights = malloc(sizeof(double) * weights_size);
            for (int k = 0; k < weights_size; k++) {
                net->hidden[l].neurons[j].weights[k] = rand_double();
            }
        }
    }

    // Initialize the Output layer
    net->output.next = NULL;
    net->output.prev = &net->hidden[net->hidden_layer_size-1];
    net->output.size = output_n;
    net->output.neurons = malloc(sizeof(neuron_t) * net->output.size);
    weights_size = net->output.prev->size;
    for (int j = 0; j < net->output.size; j++) {
        net->output.neurons[j].bias = NN_DEFAULT_BIAS;
        net->output.neurons[j].weights = malloc(sizeof(double) * weights_size);
        for (int k = 0; k < weights_size; k++) {
            net->output.neurons[j].weights[k] = rand_double();
        }
    }
}

static void update_neurons(layer_t* const layer)
{
    neuron_t *neuron;

    for (int j = 0; j < layer->size; j++) {
        neuron = &layer->neurons[j];
        neuron->weighted_input = 0;
        for (int k = 0; k < layer->prev->size; k++)
            neuron->weighted_input += layer->prev->neurons[k].activation * neuron->weights[k];
        neuron->weighted_input += neuron->bias;

        neuron->activation = activation_fun(neuron->weighted_input);
    }
}

void forward_propagate(neural_network_t *net, const double *inputs)
{
    layer_t *layer = &net->input;

    // Load the data into the input layer's neurons
    for (int j = 0; j < layer->size; j++)
        layer->neurons[j].activation = inputs[j];
    layer = layer->next;

    // Forward propagate through the hidden and output layers
    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++)
            update_neurons(layer);
        layer = layer->next;
    }
}

void back_propagate(neural_network_t *net, const double *expected)
{
    layer_t *layer = &net->output;

    // Calculate dz for the output's neurons
    for (int j = 0; j < layer->size; j++)
        layer->neurons[j].dz = cost_prime_fun(layer->neurons[j].activation, expected[j]) * activation_prime_fun(layer->neurons[j].weighted_input);
    layer = layer->prev;

    // Calculate dz for the rest of the neurons
    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++) {
            layer->neurons[j].dz = 0;
            for (int jp1 = 0; jp1 < layer->next->size; jp1++)
                layer->neurons[j].dz += layer->next->neurons[jp1].dz * layer->next->neurons[jp1].weights[j];
            layer->neurons[j].dz *= activation_prime_fun(layer->neurons[j].weighted_input);
        }
        layer = layer->prev;
    }
}

void update_weights_biases(neural_network_t *net)
{
    layer_t *layer = &net->output;

    while (layer->prev != NULL) {
        for (int j = 0; j < layer->size; j++) {
            for (int k = 0; k < layer->prev->size; k++)
                layer->neurons[j].weights[k] += NN_LEARNING_RATE * layer->neurons[j].dz * layer->prev->neurons[k].activation;
            layer->neurons[j].bias += NN_LEARNING_RATE * layer->neurons[j].dz;
        }
        layer = layer->prev;
    }
}


// HELPER FUNCTIONS //
static inline double rand_double(void)
{
    double x;
    while ((x = rand()/(double)RAND_MAX) == 0)      // Returns a random double (0, 1]
        ;
    return x;
}

// Records the current weights and values of a net's neurons for use with the visualizer
static void record(neural_network_t *net, int flag)
{
    static double max_weight = 0;
    FILE *fp = fopen(NN_VISUALIZER_PATH, "a");
    layer_t *layer = &net->input;

    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++) {
            fprintf(fp, "%.3f,", layer->neurons[j].activation);
            if (layer != &net->input) {
                for (int k = 0; k < layer->prev->size; k++) {
                    if (fabs(layer->neurons[j].weights[k]) > max_weight)
                        max_weight = fabs(layer->neurons[j].weights[k]);
                    fprintf(fp, "%.3f,", layer->neurons[j].weights[k]);
                }
            }
            fprintf(fp, "\n");
        }
        layer = layer->next;
    }
    fprintf(fp, "=,\n");
    if (flag)
        fprintf(fp, "%.3f,", max_weight);
    fclose(fp);
}


// MATH FUNCTIONS //
static double activation_fun(double input)
{
    return 1.0/(1.0 + exp(-input));
}

static double activation_prime_fun(double input)
{
    return activation_fun(input) * (1.0 - activation_fun(input));
}

static double cost_fun(double predicted, double expected)
{
    return pow((expected - predicted), 2) / 2.0;
}

static double cost_prime_fun(double predicted, double expected)
{
    return (expected - predicted);
}
