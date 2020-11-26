#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "data.h"
#include "nn.h"


static inline double* max(double *a, double *b, double *c)
{
    if (*a > *b && *a > *c)
        return a;
    if (*b > *a && *b > *c)
        return b ;
    return c;
}

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
    if (flag == 1)
        fprintf(fp, "%.3f,", max_weight);
    fclose(fp);
}

static void print_weights(neural_network_t *net)
{
    int swt = 0;
    layer_t *layer = &net->input;

    printf("\nprint weights\n===============================\n");
    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++) {
            if (layer == &net->input) {
                if (swt && j == 0) printf("\nINPUT\n");
                //printf("%.2f\n", layer->neurons[j].activation);
            } else {
                if (swt) printf("\n===================\nNEURON %d:\n", j);
                for (int k = 0; k < layer->prev->size; k++) {
                    if (swt) printf("%d: ", k);
                    printf("%.20f\n", k, layer->neurons[j].weights[k]);
                }
                if (swt) printf("b: ");
                printf("%.20f\n", layer->neurons->bias);
            }
        }
        layer = layer->next;
    }
    printf("===============================\n");
}

static void save_weights(neural_network_t *net)
{
    FILE *fp = fopen(NN_SAVE_PATH, "w");
    layer_t *layer = net->input.next;

    //printf("\nsave weights\n===============================\n");
    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++) {
            for (int k = 0; k < layer->prev->size; k++) {
                fprintf(fp, "%.20f\n", layer->neurons[j].weights[k]);
                //printf("%.20f\n", layer->neurons[j].weights[k]);
            }
            fprintf(fp, "%.20f\n", layer->neurons->bias);
            //printf("%.20f\n", layer->neurons->bias);
        }
        layer = layer->next;
    }
    fclose(fp);
    //printf("===============================\n");
}

static void load_weights(neural_network_t *net)
{
    FILE *fp = fopen(NN_SAVE_PATH, "r");
    layer_t *layer = net->input.next;
    double buf;

    //printf("\nload weights\n===============================\n");
    while (layer != NULL) {
        for (int j = 0; j < layer->size; j++) {
            for (int k = 0; k < layer->prev->size; k++) {
                fscanf(fp, "%lf", &buf);
                layer->neurons[j].weights[k] = buf;
                //printf("%.20f\n", layer->neurons[j].weights[k]);
            }
            fscanf(fp, "%lf", &buf);
            layer->neurons[j].bias = buf;
            //printf("%.20f\n", layer->neurons[j].bias);
        }
        layer = layer->next;
    }
    fclose(fp);
    //printf("===============================\n");
}


int main() {
    long seed = (long)time(NULL);
    srand(seed);

    // Initialize the data
    data_t training_data[NN_TRAINING_SIZE];
    data_t testing_data[NN_TESTING_SIZE];
    load_training_data(training_data);
    load_testing_data(testing_data);

    // Initialize the neural net
    neural_network_t nn;
    int neurons_per_layer[NN_NUM_HIDDEN_LAYERS] = {4, 6};
    init_nn(&nn, NN_NUM_INPUTS, NN_NUM_OUTPUTS, NN_NUM_HIDDEN_LAYERS, neurons_per_layer);

    if(NN_RECORD)
        remove("../visualizer/data.csv");
    if (NN_LOAD_WEIGHTS && access(NN_SAVE_PATH, F_OK) != -1) {
        load_weights(&nn);
    } else {
        for (int e = 0; e < NN_EPOCHS; e++) {
            for (int i = 0; i < NN_TRAINING_SIZE; i++) {
                forward_propagate(&nn, training_data[i].inputs);
                back_propagate(&nn, training_data[i].expected_output);
                update_weights_biases(&nn);
            }
            if (NN_RECORD) {
                if (e == NN_EPOCHS-1)
                    record(&nn, 1);
                else if (e % 10 == 0)
                    record(&nn, 0);
            }
        }
        if (NN_SAVE_WEIGHTS)
            save_weights(&nn);
    }
    print_weights(&nn);

    // Score the model
    int n_correct = 0;
    for (int i = 0; i < NN_TESTING_SIZE; i++) {
        forward_propagate(&nn, testing_data[i].inputs);
        for (int j = 0; j < nn.output.size; j++) {
            printf("%d:\t%lf\t%lf", j, nn.output.neurons[j].activation, testing_data[i].expected_output[j]);
            if (&nn.output.neurons[j].activation == max(&nn.output.neurons[0].activation, &nn.output.neurons[1].activation, &nn.output.neurons[2].activation)) {
                if (testing_data[i].expected_output[j] == 1) {
                    printf(" <<<<<");
                    n_correct++;
                } else {
                    printf(" XXX");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("===============================\n");

    printf("Seed: %ld\n", seed);
    printf("Model Accuracy: %.4f%%\n", (((double)n_correct)/NN_TESTING_SIZE) * 100.0);
    //print_weights(&nn);
}
