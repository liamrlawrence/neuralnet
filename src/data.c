#include <stdlib.h>
#include "data.h"
#include "nn.h"

static inline void shuffle(data_t *data, int size)
{
    for (int i = size-1; i > 0; i--) {
        int j = rand() % (i + 1);
        data_t tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

void load_testing_data(data_t *data)
{
    for (int i = 0; i < NN_TESTING_SIZE; i++) {
        data[i].inputs = malloc(sizeof(double) * NN_INPUTS);
        data[i].expected_output = malloc(sizeof(double) * NN_OUTPUTS);
    }

    data[0].inputs[0] = 4.6;
    data[0].inputs[1] = 3.1;
    data[0].inputs[2] = 1.5;
    data[0].inputs[3] = 0.2;
    data[0].expected_output[0] = 1;
    data[0].expected_output[1] = 0;
    data[0].expected_output[2] = 0;

    data[1].inputs[0] = 4.6;
    data[1].inputs[1] = 3.4;
    data[1].inputs[2] = 1.4;
    data[1].inputs[3] = 0.3;
    data[1].expected_output[0] = 1;
    data[1].expected_output[1] = 0;
    data[1].expected_output[2] = 0;

    data[2].inputs[0] = 5;
    data[2].inputs[1] = 3.4;
    data[2].inputs[2] = 1.5;
    data[2].inputs[3] = 0.2;
    data[2].expected_output[0] = 1;
    data[2].expected_output[1] = 0;
    data[2].expected_output[2] = 0;

    data[3].inputs[0] = 4.9;
    data[3].inputs[1] = 3.1;
    data[3].inputs[2] = 1.5;
    data[3].inputs[3] = 0.1;
    data[3].expected_output[0] = 1;
    data[3].expected_output[1] = 0;
    data[3].expected_output[2] = 0;

    data[4].inputs[0] = 5.4;
    data[4].inputs[1] = 3.7;
    data[4].inputs[2] = 1.5;
    data[4].inputs[3] = 0.2;
    data[4].expected_output[0] = 1;
    data[4].expected_output[1] = 0;
    data[4].expected_output[2] = 0;

    data[5].inputs[0] = 4.8;
    data[5].inputs[1] = 3;
    data[5].inputs[2] = 1.4;
    data[5].inputs[3] = 0.1;
    data[5].expected_output[0] = 1;
    data[5].expected_output[1] = 0;
    data[5].expected_output[2] = 0;

    data[6].inputs[0] = 5.8;
    data[6].inputs[1] = 4;
    data[6].inputs[2] = 1.2;
    data[6].inputs[3] = 0.2;
    data[6].expected_output[0] = 1;
    data[6].expected_output[1] = 0;
    data[6].expected_output[2] = 0;

    data[7].inputs[0] = 5.1;
    data[7].inputs[1] = 3.5;
    data[7].inputs[2] = 1.4;
    data[7].inputs[3] = 0.3;
    data[7].expected_output[0] = 1;
    data[7].expected_output[1] = 0;
    data[7].expected_output[2] = 0;

    data[8].inputs[0] = 5.7;
    data[8].inputs[1] = 3.8;
    data[8].inputs[2] = 1.7;
    data[8].inputs[3] = 0.3;
    data[8].expected_output[0] = 1;
    data[8].expected_output[1] = 0;
    data[8].expected_output[2] = 0;

    data[9].inputs[0] = 5.4;
    data[9].inputs[1] = 3.4;
    data[9].inputs[2] = 1.7;
    data[9].inputs[3] = 0.2;
    data[9].expected_output[0] = 1;
    data[9].expected_output[1] = 0;
    data[9].expected_output[2] = 0;

    data[10].inputs[0] = 5.1;
    data[10].inputs[1] = 3.3;
    data[10].inputs[2] = 1.7;
    data[10].inputs[3] = 0.5;
    data[10].expected_output[0] = 1;
    data[10].expected_output[1] = 0;
    data[10].expected_output[2] = 0;

    data[11].inputs[0] = 5;
    data[11].inputs[1] = 3;
    data[11].inputs[2] = 1.6;
    data[11].inputs[3] = 0.2;
    data[11].expected_output[0] = 1;
    data[11].expected_output[1] = 0;
    data[11].expected_output[2] = 0;

    data[12].inputs[0] = 4.7;
    data[12].inputs[1] = 3.2;
    data[12].inputs[2] = 1.6;
    data[12].inputs[3] = 0.2;
    data[12].expected_output[0] = 1;
    data[12].expected_output[1] = 0;
    data[12].expected_output[2] = 0;

    data[13].inputs[0] = 5.4;
    data[13].inputs[1] = 3.4;
    data[13].inputs[2] = 1.5;
    data[13].inputs[3] = 0.4;
    data[13].expected_output[0] = 1;
    data[13].expected_output[1] = 0;
    data[13].expected_output[2] = 0;

    data[14].inputs[0] = 5.1;
    data[14].inputs[1] = 3.4;
    data[14].inputs[2] = 1.5;
    data[14].inputs[3] = 0.2;
    data[14].expected_output[0] = 1;
    data[14].expected_output[1] = 0;
    data[14].expected_output[2] = 0;

    data[15].inputs[0] = 4.4;
    data[15].inputs[1] = 3.2;
    data[15].inputs[2] = 1.3;
    data[15].inputs[3] = 0.2;
    data[15].expected_output[0] = 1;
    data[15].expected_output[1] = 0;
    data[15].expected_output[2] = 0;

    data[16].inputs[0] = 5;
    data[16].inputs[1] = 3.5;
    data[16].inputs[2] = 1.6;
    data[16].inputs[3] = 0.6;
    data[16].expected_output[0] = 1;
    data[16].expected_output[1] = 0;
    data[16].expected_output[2] = 0;

    data[17].inputs[0] = 5.1;
    data[17].inputs[1] = 3.8;
    data[17].inputs[2] = 1.9;
    data[17].inputs[3] = 0.4;
    data[17].expected_output[0] = 1;
    data[17].expected_output[1] = 0;
    data[17].expected_output[2] = 0;

    data[18].inputs[0] = 4.6;
    data[18].inputs[1] = 3.2;
    data[18].inputs[2] = 1.4;
    data[18].inputs[3] = 0.2;
    data[18].expected_output[0] = 1;
    data[18].expected_output[1] = 0;
    data[18].expected_output[2] = 0;

    data[19].inputs[0] = 5.3;
    data[19].inputs[1] = 3.7;
    data[19].inputs[2] = 1.5;
    data[19].inputs[3] = 0.2;
    data[19].expected_output[0] = 1;
    data[19].expected_output[1] = 0;
    data[19].expected_output[2] = 0;

    data[20].inputs[0] = 5;
    data[20].inputs[1] = 3.3;
    data[20].inputs[2] = 1.4;
    data[20].inputs[3] = 0.2;
    data[20].expected_output[0] = 1;
    data[20].expected_output[1] = 0;
    data[20].expected_output[2] = 0;

    data[21].inputs[0] = 5.5;
    data[21].inputs[1] = 2.3;
    data[21].inputs[2] = 4;
    data[21].inputs[3] = 1.3;
    data[21].expected_output[0] = 0;
    data[21].expected_output[1] = 1;
    data[21].expected_output[2] = 0;

    data[22].inputs[0] = 5.7;
    data[22].inputs[1] = 2.8;
    data[22].inputs[2] = 4.5;
    data[22].inputs[3] = 1.3;
    data[22].expected_output[0] = 0;
    data[22].expected_output[1] = 1;
    data[22].expected_output[2] = 0;

    data[23].inputs[0] = 5.2;
    data[23].inputs[1] = 2.7;
    data[23].inputs[2] = 3.9;
    data[23].inputs[3] = 1.4;
    data[23].expected_output[0] = 0;
    data[23].expected_output[1] = 1;
    data[23].expected_output[2] = 0;

    data[24].inputs[0] = 6.1;
    data[24].inputs[1] = 2.9;
    data[24].inputs[2] = 4.7;
    data[24].inputs[3] = 1.4;
    data[24].expected_output[0] = 0;
    data[24].expected_output[1] = 1;
    data[24].expected_output[2] = 0;

    data[25].inputs[0] = 6.7;
    data[25].inputs[1] = 3.1;
    data[25].inputs[2] = 4.4;
    data[25].inputs[3] = 1.4;
    data[25].expected_output[0] = 0;
    data[25].expected_output[1] = 1;
    data[25].expected_output[2] = 0;

    data[26].inputs[0] = 5.9;
    data[26].inputs[1] = 3.2;
    data[26].inputs[2] = 4.8;
    data[26].inputs[3] = 1.8;
    data[26].expected_output[0] = 0;
    data[26].expected_output[1] = 1;
    data[26].expected_output[2] = 0;

    data[27].inputs[0] = 6.4;
    data[27].inputs[1] = 2.9;
    data[27].inputs[2] = 4.3;
    data[27].inputs[3] = 1.3;
    data[27].expected_output[0] = 0;
    data[27].expected_output[1] = 1;
    data[27].expected_output[2] = 0;

    data[28].inputs[0] = 6.6;
    data[28].inputs[1] = 3;
    data[28].inputs[2] = 4.4;
    data[28].inputs[3] = 1.4;
    data[28].expected_output[0] = 0;
    data[28].expected_output[1] = 1;
    data[28].expected_output[2] = 0;

    data[29].inputs[0] = 6;
    data[29].inputs[1] = 2.9;
    data[29].inputs[2] = 4.5;
    data[29].inputs[3] = 1.5;
    data[29].expected_output[0] = 0;
    data[29].expected_output[1] = 1;
    data[29].expected_output[2] = 0;

    data[30].inputs[0] = 5.5;
    data[30].inputs[1] = 2.4;
    data[30].inputs[2] = 3.7;
    data[30].inputs[3] = 1;
    data[30].expected_output[0] = 0;
    data[30].expected_output[1] = 1;
    data[30].expected_output[2] = 0;

    data[31].inputs[0] = 5.8;
    data[31].inputs[1] = 2.7;
    data[31].inputs[2] = 3.9;
    data[31].inputs[3] = 1.2;
    data[31].expected_output[0] = 0;
    data[31].expected_output[1] = 1;
    data[31].expected_output[2] = 0;

    data[32].inputs[0] = 5.6;
    data[32].inputs[1] = 3;
    data[32].inputs[2] = 4.1;
    data[32].inputs[3] = 1.3;
    data[32].expected_output[0] = 0;
    data[32].expected_output[1] = 1;
    data[32].expected_output[2] = 0;

    data[33].inputs[0] = 5.7;
    data[33].inputs[1] = 2.9;
    data[33].inputs[2] = 4.2;
    data[33].inputs[3] = 1.3;
    data[33].expected_output[0] = 0;
    data[33].expected_output[1] = 1;
    data[33].expected_output[2] = 0;

    data[34].inputs[0] = 5.1;
    data[34].inputs[1] = 2.5;
    data[34].inputs[2] = 3;
    data[34].inputs[3] = 1.1;
    data[34].expected_output[0] = 0;
    data[34].expected_output[1] = 1;
    data[34].expected_output[2] = 0;

    data[35].inputs[0] = 7.1;
    data[35].inputs[1] = 3;
    data[35].inputs[2] = 5.9;
    data[35].inputs[3] = 2.1;
    data[35].expected_output[0] = 0;
    data[35].expected_output[1] = 0;
    data[35].expected_output[2] = 1;

    data[36].inputs[0] = 6.3;
    data[36].inputs[1] = 2.9;
    data[36].inputs[2] = 5.6;
    data[36].inputs[3] = 1.8;
    data[36].expected_output[0] = 0;
    data[36].expected_output[1] = 0;
    data[36].expected_output[2] = 1;

    data[37].inputs[0] = 4.9;
    data[37].inputs[1] = 2.5;
    data[37].inputs[2] = 4.5;
    data[37].inputs[3] = 1.7;
    data[37].expected_output[0] = 0;
    data[37].expected_output[1] = 0;
    data[37].expected_output[2] = 1;

    data[38].inputs[0] = 5.7;
    data[38].inputs[1] = 2.5;
    data[38].inputs[2] = 5;
    data[38].inputs[3] = 2;
    data[38].expected_output[0] = 0;
    data[38].expected_output[1] = 0;
    data[38].expected_output[2] = 1;

    data[39].inputs[0] = 6.5;
    data[39].inputs[1] = 3;
    data[39].inputs[2] = 5.5;
    data[39].inputs[3] = 1.8;
    data[39].expected_output[0] = 0;
    data[39].expected_output[1] = 0;
    data[39].expected_output[2] = 1;

    data[40].inputs[0] = 7.7;
    data[40].inputs[1] = 3.8;
    data[40].inputs[2] = 6.7;
    data[40].inputs[3] = 2.2;
    data[40].expected_output[0] = 0;
    data[40].expected_output[1] = 0;
    data[40].expected_output[2] = 1;

    data[41].inputs[0] = 6.9;
    data[41].inputs[1] = 3.2;
    data[41].inputs[2] = 5.7;
    data[41].inputs[3] = 2.3;
    data[41].expected_output[0] = 0;
    data[41].expected_output[1] = 0;
    data[41].expected_output[2] = 1;

    data[42].inputs[0] = 7.7;
    data[42].inputs[1] = 2.8;
    data[42].inputs[2] = 6.7;
    data[42].inputs[3] = 2;
    data[42].expected_output[0] = 0;
    data[42].expected_output[1] = 0;
    data[42].expected_output[2] = 1;

    data[43].inputs[0] = 6.7;
    data[43].inputs[1] = 3.3;
    data[43].inputs[2] = 5.7;
    data[43].inputs[3] = 2.1;
    data[43].expected_output[0] = 0;
    data[43].expected_output[1] = 0;
    data[43].expected_output[2] = 1;

    data[44].inputs[0] = 6.2;
    data[44].inputs[1] = 2.8;
    data[44].inputs[2] = 4.8;
    data[44].inputs[3] = 1.8;
    data[44].expected_output[0] = 0;
    data[44].expected_output[1] = 0;
    data[44].expected_output[2] = 1;

    data[45].inputs[0] = 6.1;
    data[45].inputs[1] = 3;
    data[45].inputs[2] = 4.9;
    data[45].inputs[3] = 1.8;
    data[45].expected_output[0] = 0;
    data[45].expected_output[1] = 0;
    data[45].expected_output[2] = 1;

    data[46].inputs[0] = 6.4;
    data[46].inputs[1] = 2.8;
    data[46].inputs[2] = 5.6;
    data[46].inputs[3] = 2.1;
    data[46].expected_output[0] = 0;
    data[46].expected_output[1] = 0;
    data[46].expected_output[2] = 1;

    data[47].inputs[0] = 7.7;
    data[47].inputs[1] = 3;
    data[47].inputs[2] = 6.1;
    data[47].inputs[3] = 2.3;
    data[47].expected_output[0] = 0;
    data[47].expected_output[1] = 0;
    data[47].expected_output[2] = 1;

    data[48].inputs[0] = 6.7;
    data[48].inputs[1] = 3.1;
    data[48].inputs[2] = 5.6;
    data[48].inputs[3] = 2.4;
    data[48].expected_output[0] = 0;
    data[48].expected_output[1] = 0;
    data[48].expected_output[2] = 1;

    data[49].inputs[0] = 6.9;
    data[49].inputs[1] = 3.1;
    data[49].inputs[2] = 5.1;
    data[49].inputs[3] = 2.3;
    data[49].expected_output[0] = 0;
    data[49].expected_output[1] = 0;
    data[49].expected_output[2] = 1;

    shuffle(data, NN_TESTING_SIZE);
}

void load_training_data(data_t *data)
{
    for (int i = 0; i < NN_TRAINING_SIZE; i++) {
        data[i].inputs = malloc(sizeof(double) * NN_INPUTS);
        data[i].expected_output = malloc(sizeof(double) * NN_OUTPUTS);
    }

    data[0].inputs[0] = 5.1;
    data[0].inputs[1] = 3.5;
    data[0].inputs[2] = 1.4;
    data[0].inputs[3] = 0.2;
    data[0].expected_output[0] = 1;
    data[0].expected_output[1] = 0;
    data[0].expected_output[2] = 0;

    data[1].inputs[0] = 4.9;
    data[1].inputs[1] = 3;
    data[1].inputs[2] = 1.4;
    data[1].inputs[3] = 0.2;
    data[1].expected_output[0] = 1;
    data[1].expected_output[1] = 0;
    data[1].expected_output[2] = 0;

    data[2].inputs[0] = 4.7;
    data[2].inputs[1] = 3.2;
    data[2].inputs[2] = 1.3;
    data[2].inputs[3] = 0.2;
    data[2].expected_output[0] = 1;
    data[2].expected_output[1] = 0;
    data[2].expected_output[2] = 0;

    data[3].inputs[0] = 5;
    data[3].inputs[1] = 3.6;
    data[3].inputs[2] = 1.4;
    data[3].inputs[3] = 0.2;
    data[3].expected_output[0] = 1;
    data[3].expected_output[1] = 0;
    data[3].expected_output[2] = 0;

    data[4].inputs[0] = 5.4;
    data[4].inputs[1] = 3.9;
    data[4].inputs[2] = 1.7;
    data[4].inputs[3] = 0.4;
    data[4].expected_output[0] = 1;
    data[4].expected_output[1] = 0;
    data[4].expected_output[2] = 0;

    data[5].inputs[0] = 4.4;
    data[5].inputs[1] = 2.9;
    data[5].inputs[2] = 1.4;
    data[5].inputs[3] = 0.2;
    data[5].expected_output[0] = 1;
    data[5].expected_output[1] = 0;
    data[5].expected_output[2] = 0;

    data[6].inputs[0] = 4.8;
    data[6].inputs[1] = 3.4;
    data[6].inputs[2] = 1.6;
    data[6].inputs[3] = 0.2;
    data[6].expected_output[0] = 1;
    data[6].expected_output[1] = 0;
    data[6].expected_output[2] = 0;

    data[7].inputs[0] = 4.3;
    data[7].inputs[1] = 3;
    data[7].inputs[2] = 1.1;
    data[7].inputs[3] = 0.1;
    data[7].expected_output[0] = 1;
    data[7].expected_output[1] = 0;
    data[7].expected_output[2] = 0;

    data[8].inputs[0] = 5.7;
    data[8].inputs[1] = 4.4;
    data[8].inputs[2] = 1.5;
    data[8].inputs[3] = 0.4;
    data[8].expected_output[0] = 1;
    data[8].expected_output[1] = 0;
    data[8].expected_output[2] = 0;

    data[9].inputs[0] = 5.4;
    data[9].inputs[1] = 3.9;
    data[9].inputs[2] = 1.3;
    data[9].inputs[3] = 0.4;
    data[9].expected_output[0] = 1;
    data[9].expected_output[1] = 0;
    data[9].expected_output[2] = 0;

    data[10].inputs[0] = 5.1;
    data[10].inputs[1] = 3.8;
    data[10].inputs[2] = 1.5;
    data[10].inputs[3] = 0.3;
    data[10].expected_output[0] = 1;
    data[10].expected_output[1] = 0;
    data[10].expected_output[2] = 0;

    data[11].inputs[0] = 5.1;
    data[11].inputs[1] = 3.7;
    data[11].inputs[2] = 1.5;
    data[11].inputs[3] = 0.4;
    data[11].expected_output[0] = 1;
    data[11].expected_output[1] = 0;
    data[11].expected_output[2] = 0;

    data[12].inputs[0] = 4.6;
    data[12].inputs[1] = 3.6;
    data[12].inputs[2] = 1;
    data[12].inputs[3] = 0.2;
    data[12].expected_output[0] = 1;
    data[12].expected_output[1] = 0;
    data[12].expected_output[2] = 0;

    data[13].inputs[0] = 4.8;
    data[13].inputs[1] = 3.4;
    data[13].inputs[2] = 1.9;
    data[13].inputs[3] = 0.2;
    data[13].expected_output[0] = 1;
    data[13].expected_output[1] = 0;
    data[13].expected_output[2] = 0;

    data[14].inputs[0] = 5;
    data[14].inputs[1] = 3.4;
    data[14].inputs[2] = 1.6;
    data[14].inputs[3] = 0.4;
    data[14].expected_output[0] = 1;
    data[14].expected_output[1] = 0;
    data[14].expected_output[2] = 0;

    data[15].inputs[0] = 5.2;
    data[15].inputs[1] = 3.5;
    data[15].inputs[2] = 1.5;
    data[15].inputs[3] = 0.2;
    data[15].expected_output[0] = 1;
    data[15].expected_output[1] = 0;
    data[15].expected_output[2] = 0;

    data[16].inputs[0] = 5.2;
    data[16].inputs[1] = 3.4;
    data[16].inputs[2] = 1.4;
    data[16].inputs[3] = 0.2;
    data[16].expected_output[0] = 1;
    data[16].expected_output[1] = 0;
    data[16].expected_output[2] = 0;

    data[17].inputs[0] = 4.8;
    data[17].inputs[1] = 3.1;
    data[17].inputs[2] = 1.6;
    data[17].inputs[3] = 0.2;
    data[17].expected_output[0] = 1;
    data[17].expected_output[1] = 0;
    data[17].expected_output[2] = 0;

    data[18].inputs[0] = 5.2;
    data[18].inputs[1] = 4.1;
    data[18].inputs[2] = 1.5;
    data[18].inputs[3] = 0.1;
    data[18].expected_output[0] = 1;
    data[18].expected_output[1] = 0;
    data[18].expected_output[2] = 0;

    data[19].inputs[0] = 5.5;
    data[19].inputs[1] = 4.2;
    data[19].inputs[2] = 1.4;
    data[19].inputs[3] = 0.2;
    data[19].expected_output[0] = 1;
    data[19].expected_output[1] = 0;
    data[19].expected_output[2] = 0;

    data[20].inputs[0] = 4.9;
    data[20].inputs[1] = 3.1;
    data[20].inputs[2] = 1.5;
    data[20].inputs[3] = 0.1;
    data[20].expected_output[0] = 1;
    data[20].expected_output[1] = 0;
    data[20].expected_output[2] = 0;

    data[21].inputs[0] = 5;
    data[21].inputs[1] = 3.2;
    data[21].inputs[2] = 1.2;
    data[21].inputs[3] = 0.2;
    data[21].expected_output[0] = 1;
    data[21].expected_output[1] = 0;
    data[21].expected_output[2] = 0;

    data[22].inputs[0] = 5.5;
    data[22].inputs[1] = 3.5;
    data[22].inputs[2] = 1.3;
    data[22].inputs[3] = 0.2;
    data[22].expected_output[0] = 1;
    data[22].expected_output[1] = 0;
    data[22].expected_output[2] = 0;

    data[23].inputs[0] = 4.9;
    data[23].inputs[1] = 3.1;
    data[23].inputs[2] = 1.5;
    data[23].inputs[3] = 0.1;
    data[23].expected_output[0] = 1;
    data[23].expected_output[1] = 0;
    data[23].expected_output[2] = 0;

    data[24].inputs[0] = 4.4;
    data[24].inputs[1] = 3;
    data[24].inputs[2] = 1.3;
    data[24].inputs[3] = 0.2;
    data[24].expected_output[0] = 1;
    data[24].expected_output[1] = 0;
    data[24].expected_output[2] = 0;

    data[25].inputs[0] = 5;
    data[25].inputs[1] = 3.5;
    data[25].inputs[2] = 1.3;
    data[25].inputs[3] = 0.3;
    data[25].expected_output[0] = 1;
    data[25].expected_output[1] = 0;
    data[25].expected_output[2] = 0;

    data[26].inputs[0] = 4.5;
    data[26].inputs[1] = 2.3;
    data[26].inputs[2] = 1.3;
    data[26].inputs[3] = 0.3;
    data[26].expected_output[0] = 1;
    data[26].expected_output[1] = 0;
    data[26].expected_output[2] = 0;

    data[27].inputs[0] = 4.8;
    data[27].inputs[1] = 3;
    data[27].inputs[2] = 1.4;
    data[27].inputs[3] = 0.3;
    data[27].expected_output[0] = 1;
    data[27].expected_output[1] = 0;
    data[27].expected_output[2] = 0;

    data[28].inputs[0] = 5.1;
    data[28].inputs[1] = 3.8;
    data[28].inputs[2] = 1.6;
    data[28].inputs[3] = 0.2;
    data[28].expected_output[0] = 1;
    data[28].expected_output[1] = 0;
    data[28].expected_output[2] = 0;

    data[29].inputs[0] = 7;
    data[29].inputs[1] = 3.2;
    data[29].inputs[2] = 4.7;
    data[29].inputs[3] = 1.4;
    data[29].expected_output[0] = 0;
    data[29].expected_output[1] = 1;
    data[29].expected_output[2] = 0;

    data[30].inputs[0] = 6.4;
    data[30].inputs[1] = 3.2;
    data[30].inputs[2] = 4.5;
    data[30].inputs[3] = 1.5;
    data[30].expected_output[0] = 0;
    data[30].expected_output[1] = 1;
    data[30].expected_output[2] = 0;

    data[31].inputs[0] = 6.9;
    data[31].inputs[1] = 3.1;
    data[31].inputs[2] = 4.9;
    data[31].inputs[3] = 1.5;
    data[31].expected_output[0] = 0;
    data[31].expected_output[1] = 1;
    data[31].expected_output[2] = 0;

    data[32].inputs[0] = 6.5;
    data[32].inputs[1] = 2.8;
    data[32].inputs[2] = 4.6;
    data[32].inputs[3] = 1.5;
    data[32].expected_output[0] = 0;
    data[32].expected_output[1] = 1;
    data[32].expected_output[2] = 0;

    data[33].inputs[0] = 6.3;
    data[33].inputs[1] = 3.3;
    data[33].inputs[2] = 4.7;
    data[33].inputs[3] = 1.6;
    data[33].expected_output[0] = 0;
    data[33].expected_output[1] = 1;
    data[33].expected_output[2] = 0;

    data[34].inputs[0] = 4.9;
    data[34].inputs[1] = 2.4;
    data[34].inputs[2] = 3.3;
    data[34].inputs[3] = 1;
    data[34].expected_output[0] = 0;
    data[34].expected_output[1] = 1;
    data[34].expected_output[2] = 0;

    data[35].inputs[0] = 6.6;
    data[35].inputs[1] = 2.9;
    data[35].inputs[2] = 4.6;
    data[35].inputs[3] = 1.3;
    data[35].expected_output[0] = 0;
    data[35].expected_output[1] = 1;
    data[35].expected_output[2] = 0;

    data[36].inputs[0] = 5;
    data[36].inputs[1] = 2;
    data[36].inputs[2] = 3.5;
    data[36].inputs[3] = 1;
    data[36].expected_output[0] = 0;
    data[36].expected_output[1] = 1;
    data[36].expected_output[2] = 0;

    data[37].inputs[0] = 5.9;
    data[37].inputs[1] = 3;
    data[37].inputs[2] = 4.2;
    data[37].inputs[3] = 1.5;
    data[37].expected_output[0] = 0;
    data[37].expected_output[1] = 1;
    data[37].expected_output[2] = 0;

    data[38].inputs[0] = 6;
    data[38].inputs[1] = 2.2;
    data[38].inputs[2] = 4;
    data[38].inputs[3] = 1;
    data[38].expected_output[0] = 0;
    data[38].expected_output[1] = 1;
    data[38].expected_output[2] = 0;

    data[39].inputs[0] = 5.6;
    data[39].inputs[1] = 2.9;
    data[39].inputs[2] = 3.6;
    data[39].inputs[3] = 1.3;
    data[39].expected_output[0] = 0;
    data[39].expected_output[1] = 1;
    data[39].expected_output[2] = 0;

    data[40].inputs[0] = 5.6;
    data[40].inputs[1] = 3;
    data[40].inputs[2] = 4.5;
    data[40].inputs[3] = 1.5;
    data[40].expected_output[0] = 0;
    data[40].expected_output[1] = 1;
    data[40].expected_output[2] = 0;

    data[41].inputs[0] = 5.8;
    data[41].inputs[1] = 2.7;
    data[41].inputs[2] = 4.1;
    data[41].inputs[3] = 1;
    data[41].expected_output[0] = 0;
    data[41].expected_output[1] = 1;
    data[41].expected_output[2] = 0;

    data[42].inputs[0] = 6.2;
    data[42].inputs[1] = 2.2;
    data[42].inputs[2] = 4.5;
    data[42].inputs[3] = 1.5;
    data[42].expected_output[0] = 0;
    data[42].expected_output[1] = 1;
    data[42].expected_output[2] = 0;

    data[43].inputs[0] = 5.6;
    data[43].inputs[1] = 2.5;
    data[43].inputs[2] = 3.9;
    data[43].inputs[3] = 1.1;
    data[43].expected_output[0] = 0;
    data[43].expected_output[1] = 1;
    data[43].expected_output[2] = 0;

    data[44].inputs[0] = 6.1;
    data[44].inputs[1] = 2.8;
    data[44].inputs[2] = 4;
    data[44].inputs[3] = 1.3;
    data[44].expected_output[0] = 0;
    data[44].expected_output[1] = 1;
    data[44].expected_output[2] = 0;

    data[45].inputs[0] = 6.3;
    data[45].inputs[1] = 2.5;
    data[45].inputs[2] = 4.9;
    data[45].inputs[3] = 1.5;
    data[45].expected_output[0] = 0;
    data[45].expected_output[1] = 1;
    data[45].expected_output[2] = 0;

    data[46].inputs[0] = 6.1;
    data[46].inputs[1] = 2.8;
    data[46].inputs[2] = 4.7;
    data[46].inputs[3] = 1.2;
    data[46].expected_output[0] = 0;
    data[46].expected_output[1] = 1;
    data[46].expected_output[2] = 0;

    data[47].inputs[0] = 6.8;
    data[47].inputs[1] = 2.8;
    data[47].inputs[2] = 4.8;
    data[47].inputs[3] = 1.4;
    data[47].expected_output[0] = 0;
    data[47].expected_output[1] = 1;
    data[47].expected_output[2] = 0;

    data[48].inputs[0] = 6.7;
    data[48].inputs[1] = 3;
    data[48].inputs[2] = 5;
    data[48].inputs[3] = 1.7;
    data[48].expected_output[0] = 0;
    data[48].expected_output[1] = 1;
    data[48].expected_output[2] = 0;

    data[49].inputs[0] = 5.7;
    data[49].inputs[1] = 2.6;
    data[49].inputs[2] = 3.5;
    data[49].inputs[3] = 1;
    data[49].expected_output[0] = 0;
    data[49].expected_output[1] = 1;
    data[49].expected_output[2] = 0;

    data[50].inputs[0] = 5.5;
    data[50].inputs[1] = 2.4;
    data[50].inputs[2] = 3.8;
    data[50].inputs[3] = 1.1;
    data[50].expected_output[0] = 0;
    data[50].expected_output[1] = 1;
    data[50].expected_output[2] = 0;

    data[51].inputs[0] = 6;
    data[51].inputs[1] = 2.7;
    data[51].inputs[2] = 5.1;
    data[51].inputs[3] = 1.6;
    data[51].expected_output[0] = 0;
    data[51].expected_output[1] = 1;
    data[51].expected_output[2] = 0;

    data[52].inputs[0] = 5.4;
    data[52].inputs[1] = 3;
    data[52].inputs[2] = 4.5;
    data[52].inputs[3] = 1.5;
    data[52].expected_output[0] = 0;
    data[52].expected_output[1] = 1;
    data[52].expected_output[2] = 0;

    data[53].inputs[0] = 6;
    data[53].inputs[1] = 3.4;
    data[53].inputs[2] = 4.5;
    data[53].inputs[3] = 1.6;
    data[53].expected_output[0] = 0;
    data[53].expected_output[1] = 1;
    data[53].expected_output[2] = 0;

    data[54].inputs[0] = 6.7;
    data[54].inputs[1] = 3.1;
    data[54].inputs[2] = 4.7;
    data[54].inputs[3] = 1.5;
    data[54].expected_output[0] = 0;
    data[54].expected_output[1] = 1;
    data[54].expected_output[2] = 0;

    data[55].inputs[0] = 6.3;
    data[55].inputs[1] = 2.3;
    data[55].inputs[2] = 4.4;
    data[55].inputs[3] = 1.3;
    data[55].expected_output[0] = 0;
    data[55].expected_output[1] = 1;
    data[55].expected_output[2] = 0;

    data[56].inputs[0] = 5.5;
    data[56].inputs[1] = 2.5;
    data[56].inputs[2] = 4;
    data[56].inputs[3] = 1.3;
    data[56].expected_output[0] = 0;
    data[56].expected_output[1] = 1;
    data[56].expected_output[2] = 0;

    data[57].inputs[0] = 5.5;
    data[57].inputs[1] = 2.6;
    data[57].inputs[2] = 4.4;
    data[57].inputs[3] = 1.2;
    data[57].expected_output[0] = 0;
    data[57].expected_output[1] = 1;
    data[57].expected_output[2] = 0;

    data[58].inputs[0] = 6.1;
    data[58].inputs[1] = 3;
    data[58].inputs[2] = 4.6;
    data[58].inputs[3] = 1.4;
    data[58].expected_output[0] = 0;
    data[58].expected_output[1] = 1;
    data[58].expected_output[2] = 0;

    data[59].inputs[0] = 5.8;
    data[59].inputs[1] = 2.6;
    data[59].inputs[2] = 4;
    data[59].inputs[3] = 1.2;
    data[59].expected_output[0] = 0;
    data[59].expected_output[1] = 1;
    data[59].expected_output[2] = 0;

    data[60].inputs[0] = 5;
    data[60].inputs[1] = 2.3;
    data[60].inputs[2] = 3.3;
    data[60].inputs[3] = 1;
    data[60].expected_output[0] = 0;
    data[60].expected_output[1] = 1;
    data[60].expected_output[2] = 0;

    data[61].inputs[0] = 5.6;
    data[61].inputs[1] = 2.7;
    data[61].inputs[2] = 4.2;
    data[61].inputs[3] = 1.3;
    data[61].expected_output[0] = 0;
    data[61].expected_output[1] = 1;
    data[61].expected_output[2] = 0;

    data[62].inputs[0] = 5.7;
    data[62].inputs[1] = 3;
    data[62].inputs[2] = 4.2;
    data[62].inputs[3] = 1.2;
    data[62].expected_output[0] = 0;
    data[62].expected_output[1] = 1;
    data[62].expected_output[2] = 0;

    data[63].inputs[0] = 6.2;
    data[63].inputs[1] = 2.9;
    data[63].inputs[2] = 4.3;
    data[63].inputs[3] = 1.3;
    data[63].expected_output[0] = 0;
    data[63].expected_output[1] = 1;
    data[63].expected_output[2] = 0;

    data[64].inputs[0] = 5.7;
    data[64].inputs[1] = 2.8;
    data[64].inputs[2] = 4.1;
    data[64].inputs[3] = 1.3;
    data[64].expected_output[0] = 0;
    data[64].expected_output[1] = 1;
    data[64].expected_output[2] = 0;

    data[65].inputs[0] = 6.3;
    data[65].inputs[1] = 3.3;
    data[65].inputs[2] = 6;
    data[65].inputs[3] = 2.5;
    data[65].expected_output[0] = 0;
    data[65].expected_output[1] = 0;
    data[65].expected_output[2] = 1;

    data[66].inputs[0] = 5.8;
    data[66].inputs[1] = 2.7;
    data[66].inputs[2] = 5.1;
    data[66].inputs[3] = 1.9;
    data[66].expected_output[0] = 0;
    data[66].expected_output[1] = 0;
    data[66].expected_output[2] = 1;

    data[67].inputs[0] = 6.5;
    data[67].inputs[1] = 3;
    data[67].inputs[2] = 5.8;
    data[67].inputs[3] = 2.2;
    data[67].expected_output[0] = 0;
    data[67].expected_output[1] = 0;
    data[67].expected_output[2] = 1;

    data[68].inputs[0] = 7.6;
    data[68].inputs[1] = 3;
    data[68].inputs[2] = 6.6;
    data[68].inputs[3] = 2.1;
    data[68].expected_output[0] = 0;
    data[68].expected_output[1] = 0;
    data[68].expected_output[2] = 1;

    data[69].inputs[0] = 7.3;
    data[69].inputs[1] = 2.9;
    data[69].inputs[2] = 6.3;
    data[69].inputs[3] = 1.8;
    data[69].expected_output[0] = 0;
    data[69].expected_output[1] = 0;
    data[69].expected_output[2] = 1;

    data[70].inputs[0] = 6.7;
    data[70].inputs[1] = 2.5;
    data[70].inputs[2] = 5.8;
    data[70].inputs[3] = 1.8;
    data[70].expected_output[0] = 0;
    data[70].expected_output[1] = 0;
    data[70].expected_output[2] = 1;

    data[71].inputs[0] = 7.2;
    data[71].inputs[1] = 3.6;
    data[71].inputs[2] = 6.1;
    data[71].inputs[3] = 2.5;
    data[71].expected_output[0] = 0;
    data[71].expected_output[1] = 0;
    data[71].expected_output[2] = 1;

    data[72].inputs[0] = 6.5;
    data[72].inputs[1] = 3.2;
    data[72].inputs[2] = 5.1;
    data[72].inputs[3] = 2;
    data[72].expected_output[0] = 0;
    data[72].expected_output[1] = 0;
    data[72].expected_output[2] = 1;

    data[73].inputs[0] = 6.4;
    data[73].inputs[1] = 2.7;
    data[73].inputs[2] = 5.3;
    data[73].inputs[3] = 1.9;
    data[73].expected_output[0] = 0;
    data[73].expected_output[1] = 0;
    data[73].expected_output[2] = 1;

    data[74].inputs[0] = 6.8;
    data[74].inputs[1] = 3;
    data[74].inputs[2] = 5.5;
    data[74].inputs[3] = 2.1;
    data[74].expected_output[0] = 0;
    data[74].expected_output[1] = 0;
    data[74].expected_output[2] = 1;

    data[75].inputs[0] = 5.8;
    data[75].inputs[1] = 2.8;
    data[75].inputs[2] = 5.1;
    data[75].inputs[3] = 2.4;
    data[75].expected_output[0] = 0;
    data[75].expected_output[1] = 0;
    data[75].expected_output[2] = 1;

    data[76].inputs[0] = 6.4;
    data[76].inputs[1] = 3.2;
    data[76].inputs[2] = 5.3;
    data[76].inputs[3] = 2.3;
    data[76].expected_output[0] = 0;
    data[76].expected_output[1] = 0;
    data[76].expected_output[2] = 1;

    data[77].inputs[0] = 7.7;
    data[77].inputs[1] = 2.6;
    data[77].inputs[2] = 6.9;
    data[77].inputs[3] = 2.3;
    data[77].expected_output[0] = 0;
    data[77].expected_output[1] = 0;
    data[77].expected_output[2] = 1;

    data[78].inputs[0] = 6;
    data[78].inputs[1] = 2.2;
    data[78].inputs[2] = 5;
    data[78].inputs[3] = 1.5;
    data[78].expected_output[0] = 0;
    data[78].expected_output[1] = 0;
    data[78].expected_output[2] = 1;

    data[79].inputs[0] = 5.6;
    data[79].inputs[1] = 2.8;
    data[79].inputs[2] = 4.9;
    data[79].inputs[3] = 2;
    data[79].expected_output[0] = 0;
    data[79].expected_output[1] = 0;
    data[79].expected_output[2] = 1;

    data[80].inputs[0] = 6.3;
    data[80].inputs[1] = 2.7;
    data[80].inputs[2] = 4.9;
    data[80].inputs[3] = 1.8;
    data[80].expected_output[0] = 0;
    data[80].expected_output[1] = 0;
    data[80].expected_output[2] = 1;

    data[81].inputs[0] = 7.2;
    data[81].inputs[1] = 3.2;
    data[81].inputs[2] = 6;
    data[81].inputs[3] = 1.8;
    data[81].expected_output[0] = 0;
    data[81].expected_output[1] = 0;
    data[81].expected_output[2] = 1;

    data[82].inputs[0] = 7.2;
    data[82].inputs[1] = 3;
    data[82].inputs[2] = 5.8;
    data[82].inputs[3] = 1.6;
    data[82].expected_output[0] = 0;
    data[82].expected_output[1] = 0;
    data[82].expected_output[2] = 1;

    data[83].inputs[0] = 7.4;
    data[83].inputs[1] = 2.8;
    data[83].inputs[2] = 6.1;
    data[83].inputs[3] = 1.9;
    data[83].expected_output[0] = 0;
    data[83].expected_output[1] = 0;
    data[83].expected_output[2] = 1;

    data[84].inputs[0] = 7.9;
    data[84].inputs[1] = 3.8;
    data[84].inputs[2] = 6.4;
    data[84].inputs[3] = 2;
    data[84].expected_output[0] = 0;
    data[84].expected_output[1] = 0;
    data[84].expected_output[2] = 1;

    data[85].inputs[0] = 6.4;
    data[85].inputs[1] = 2.8;
    data[85].inputs[2] = 5.6;
    data[85].inputs[3] = 2.2;
    data[85].expected_output[0] = 0;
    data[85].expected_output[1] = 0;
    data[85].expected_output[2] = 1;

    data[86].inputs[0] = 6.3;
    data[86].inputs[1] = 2.8;
    data[86].inputs[2] = 5.1;
    data[86].inputs[3] = 1.5;
    data[86].expected_output[0] = 0;
    data[86].expected_output[1] = 0;
    data[86].expected_output[2] = 1;

    data[87].inputs[0] = 6.1;
    data[87].inputs[1] = 2.6;
    data[87].inputs[2] = 5.6;
    data[87].inputs[3] = 1.4;
    data[87].expected_output[0] = 0;
    data[87].expected_output[1] = 0;
    data[87].expected_output[2] = 1;

    data[88].inputs[0] = 6.3;
    data[88].inputs[1] = 3.4;
    data[88].inputs[2] = 5.6;
    data[88].inputs[3] = 2.4;
    data[88].expected_output[0] = 0;
    data[88].expected_output[1] = 0;
    data[88].expected_output[2] = 1;

    data[89].inputs[0] = 6.4;
    data[89].inputs[1] = 3.1;
    data[89].inputs[2] = 5.5;
    data[89].inputs[3] = 1.8;
    data[89].expected_output[0] = 0;
    data[89].expected_output[1] = 0;
    data[89].expected_output[2] = 1;

    data[90].inputs[0] = 6;
    data[90].inputs[1] = 3;
    data[90].inputs[2] = 4.8;
    data[90].inputs[3] = 1.8;
    data[90].expected_output[0] = 0;
    data[90].expected_output[1] = 0;
    data[90].expected_output[2] = 1;

    data[91].inputs[0] = 6.9;
    data[91].inputs[1] = 3.1;
    data[91].inputs[2] = 5.4;
    data[91].inputs[3] = 2.1;
    data[91].expected_output[0] = 0;
    data[91].expected_output[1] = 0;
    data[91].expected_output[2] = 1;

    data[92].inputs[0] = 6.8;
    data[92].inputs[1] = 3.2;
    data[92].inputs[2] = 5.9;
    data[92].inputs[3] = 2.3;
    data[92].expected_output[0] = 0;
    data[92].expected_output[1] = 0;
    data[92].expected_output[2] = 1;

    data[93].inputs[0] = 6.7;
    data[93].inputs[1] = 3.3;
    data[93].inputs[2] = 5.7;
    data[93].inputs[3] = 2.5;
    data[93].expected_output[0] = 0;
    data[93].expected_output[1] = 0;
    data[93].expected_output[2] = 1;

    data[94].inputs[0] = 6.7;
    data[94].inputs[1] = 3;
    data[94].inputs[2] = 5.2;
    data[94].inputs[3] = 2.3;
    data[94].expected_output[0] = 0;
    data[94].expected_output[1] = 0;
    data[94].expected_output[2] = 1;

    data[95].inputs[0] = 6.3;
    data[95].inputs[1] = 2.5;
    data[95].inputs[2] = 5;
    data[95].inputs[3] = 1.9;
    data[95].expected_output[0] = 0;
    data[95].expected_output[1] = 0;
    data[95].expected_output[2] = 1;

    data[96].inputs[0] = 5.9;
    data[96].inputs[1] = 3;
    data[96].inputs[2] = 5.1;
    data[96].inputs[3] = 1.8;
    data[96].expected_output[0] = 0;
    data[96].expected_output[1] = 0;
    data[96].expected_output[2] = 1;

    data[97].inputs[0] = 5.8;
    data[97].inputs[1] = 2.7;
    data[97].inputs[2] = 5.1;
    data[97].inputs[3] = 1.9;
    data[97].expected_output[0] = 0;
    data[97].expected_output[1] = 0;
    data[97].expected_output[2] = 1;

    data[98].inputs[0] = 6.5;
    data[98].inputs[1] = 3;
    data[98].inputs[2] = 5.2;
    data[98].inputs[3] = 2;
    data[98].expected_output[0] = 0;
    data[98].expected_output[1] = 0;
    data[98].expected_output[2] = 1;

    data[99].inputs[0] = 6.2;
    data[99].inputs[1] = 3.4;
    data[99].inputs[2] = 5.4;
    data[99].inputs[3] = 2.3;
    data[99].expected_output[0] = 0;
    data[99].expected_output[1] = 0;
    data[99].expected_output[2] = 1;

    shuffle(data, NN_TRAINING_SIZE);
}
