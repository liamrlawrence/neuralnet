#ifndef NEURALNET_DATA_H
#define NEURALNET_DATA_H

typedef struct {
    double *inputs;
    double *expected_output;
} data_t;

void load_training_data(data_t *data);
void load_testing_data(data_t *data);

#endif //NEURALNET_DATA_H
