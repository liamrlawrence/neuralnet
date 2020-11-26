class neuron {
  int x, y, n_weights;
  float val;
  float[] weights;

  neuron(int x, int y, int n_weights) {
    this.x = x;
    this.y = y;
    this.n_weights = n_weights;
    this.weights = new float[n_weights];
    //this.val = random(3);
  }

  void draw() {
    fill(255);
    strokeWeight(3);
    stroke(0);
    circle(this.x, this.y, NEURON_SIZE);

    fill(0);
    textSize(18);
    text(this.val, this.x-34, this.y+8);
  }
}
