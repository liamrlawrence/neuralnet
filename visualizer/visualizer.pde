int bgc = 250;          // background color
int NUM_INPUT = 4;
int NUM_OUTPUT = 3;
int NUM_HIDDEN_LAYERS = 2;
int NUM_LAYERS = NUM_HIDDEN_LAYERS+2;
int[] NEURONS_PER_LAYER = {NUM_INPUT, 4, 6, NUM_OUTPUT};
int NEURON_SIZE = 100;
neuron[][] nn = new neuron[NUM_LAYERS][];
Table table;
float max_weight;      // Used for normalizing the strokeWeight of the connections
int data_offset_size;  // How many rows are in each set of data
int data_offset = 0;   // The current data offset
int max_data_offset;
boolean run_mode = false;
int run_speed = 1;

void init_neurons() {
  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    // Location of the neuron
    int y_spacing = (height - NEURON_SIZE*NEURONS_PER_LAYER[layer])/(NEURONS_PER_LAYER[layer]+1);
    int x_spacing = (width - NEURON_SIZE*NUM_LAYERS)/(NUM_LAYERS);
    int y_buffer = (NEURON_SIZE/2)+y_spacing;
    int x_buffer = (x_spacing)*(layer+1);

    // Number of weights to the neuron
    int prev_num_neurons;
    if (layer == 0)
      prev_num_neurons = 0;
    else
      prev_num_neurons = NEURONS_PER_LAYER[layer-1];

    // Initialize the neuron
    for (int i = 0; i < NEURONS_PER_LAYER[layer]; i++) {
      nn[layer][i] = new neuron(x_buffer, y_buffer, prev_num_neurons);
      y_buffer += y_spacing + NEURON_SIZE;
    }
  }
}

void draw_layers() {
  int red = 0xFE;
  int green = 0x00;
  int blue = 0x00;
  int sw = 10;

  // Draw the connections
  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    for (int i = 0; i < NEURONS_PER_LAYER[layer]; i++) {
      if (layer > 0) {
        for (int w = 0; w < NEURONS_PER_LAYER[layer-1]; w++) {
          stroke(red*nn[layer][i].weights[w], green*nn[layer][i].weights[w], blue*nn[layer][i].weights[w]);
          strokeWeight((sw*nn[layer][i].weights[w])/max_weight);
          line(nn[layer][i].x, nn[layer][i].y, nn[layer-1][w].x, nn[layer-1][w].y);
        }
      }
    }
  }

  // Draw the neurons
  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    for (int i = 0; i < NEURONS_PER_LAYER[layer]; i++) {
      nn[layer][i].draw();
    }
  }
}


void setup() {
  size(1280, 720);
  background(bgc);
  table = loadTable("../data/visualizer.csv");

  // Calculate the offset size and max offset
  while (!table.getString(data_offset_size++, 0).equals("="))
    ;
  max_data_offset = (table.getRowCount() / data_offset_size)-1;
  max_weight = table.getFloat(table.getRowCount()-1, 0);
  println(max_weight);

  // Create and initialize the array of layers of neurons
  for (int layer = 0; layer < NUM_LAYERS; layer++)
    nn[layer] = new neuron[NEURONS_PER_LAYER[layer]];
  init_neurons();
}


void draw() {
  int row_counter = 0;
  int layer = 0;

  if (run_mode)
    data_offset = (data_offset+run_speed > max_data_offset) ? max_data_offset : data_offset+run_speed;

  while (!table.getString(row_counter+data_offset*data_offset_size, 0).equals("=")) {
    for (int neuron = 0; neuron < NEURONS_PER_LAYER[layer]; neuron++, row_counter++) {
      nn[layer][neuron].val = table.getFloat(row_counter+data_offset*data_offset_size, 0);
      if (layer != 0) {
        for (int w = 0; w < NEURONS_PER_LAYER[layer-1]; w++) {
          nn[layer][neuron].weights[w] = abs(table.getFloat(row_counter+data_offset*data_offset_size, w+1));
        }
      }
    }
    layer++;
  }

  background(bgc);
  fill(0);
  textSize(32);
  text(data_offset+1, width-200, height-100);

  draw_layers();
}

void keyPressed() {
  if (keyCode == RIGHT) {
    data_offset = (data_offset+1 > max_data_offset) ? max_data_offset : data_offset+1;
  } else if (keyCode == LEFT) {
    data_offset = (data_offset-1 < 0) ? 0 : data_offset-1;
  } else if (keyCode == ENTER) {
    run_mode = !run_mode;
  } else if (keyCode == SHIFT) {
    run_mode = false;
    data_offset = 0;
    run_speed = 2;
  } else if (keyCode == UP) {
    run_speed++;
  } else if (keyCode == DOWN) {
    run_speed = (run_speed-1 < 1) ? 1 : run_speed-1;
  }
}
