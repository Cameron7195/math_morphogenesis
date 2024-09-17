# A Mathematical Model of Morphogenesis

## Overview of the model
This repository outlines the code for "A Mathematical Model of Morphogenesis," a 
MASc Thesis by Cameron Witkowski, and supervised by Stephen Brown and Kevin Truong.

### Object definitions
A cell is defined as a vector, x, \in R^n.
An organism is defined as a collection of cells, represented by
X = [x1, x2, ..., xm] \in R^mxn

x and X are functions of t, ie. x(t), X(t).

Initially, an organism starts as a single cell, thus X(0) = [x1]

#### Reserved cell components
The first 3 cell components (X[:, :, 0:3]) are reserved for cell positions.

The next 3 cell components (X[:, :, 3:6]) are reserved for cell velocities.

In future, other components of the cell may be reserved for useful features,
such as mass, or some kind of death signal.

### Function definitions
Three functions define the time evolution of an organism: f, b and d.

f: R^mxn -> R^mxn
defines the changes to an organism and its cell states over time. f takes
as input the entire organism, and outputs dX/dt. ie. f(X) = dX/dt.

b: R^n -> {0, 1}
defines the birth (cell division) function. Returns 1 if a cell should divide
and 0 if the cell should do nothing.

d: R^n -> {0, 1}
defines the cell death function. Returns 1 if a cell should die and 0 if
the cell should do nothing.

### Completeness conjecture
These three functions, paired with the set of initial vectors x1, cover
the entire set of possible organismal structures.

Corollaries:

There exists an O = {x1, f, b, d} that generates a California redwood.

There exists an O = {x1, f, b, d} that generates a beating, human heart.

## Code structure

### bptt_train.py
is the main train loop used to find parameters for f.
Here, f is parametrized by a neural network utilizing the equiformer architecture.
The training loop is a simple trajectory rollout + backpropagation through
time (BPTT) loop. Currently, b is simply set to divide every 80 timesteps
until we get to 32 cells (see divide method in Organism_equiformer.py). And
no cells die currently. This initial implementation serves as a baseline for
future work.

Usage:
```python3 main_bptt.py
```

### gen_animation.py
creates the visualization (animation) using manim. Ensure the filename matches
the saved model in the 'load_model' line, e.g.:
```f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model.pt"))
```
Also ensure that the hyperparameters match the hyperparameters used to train the
model. The animation is constructed by utilizing a manim 'Graph' object. Videos
are saved to the media/videos/gen_animation/480p15/ directory by default.

Usage:
```manim -pql gen_animation.py GraphExample -o output_movie_name.mp4
```

### Organism_equiformer.py
defines the Organism class and the equiformer model, specifying how they interact.
The equiformer outputs updates to cell states and cell forces.

The Organism.sphere_loss function defines the objective used to train the neural
network. Currently, this function computes the squared difference between each cell's
distance from the origin, and some desired distance. It also penalizes neighbors that
are too close or too far.

## Setup instructions:

### Step 1: create a python virtual environment and install requirements.
1. Open a terminal and cd to this directory.
2. Run the following command to create a python virtual environment:
```python3 -m venv venv
```
3. Run the following command to activate the virtual environment:
```source venv/bin/activate
```
4. Run the following commands to install necessary requirements:
```pip3 install -r requirements.txt
```
    4.1 If pycairo fails to install, consult stackoverflow. It can be
    a little annoying sometimes, depending on OS.
5. Train a model:
```python3 bptt_train.py
```
6. Create a .mp4 file to simulate the trained organism!
```manim -pql gen_animation.py GraphExample -o simulation_1.mp4
```

## Citations

Equiformer:
https://arxiv.org/abs/2206.11990
https://github.com/lucidrains/equiformer-pytorch?tab=readme-ov-file

Manim:
https://github.com/3b1b/manim

Growing NCA (inspiration):
https://distill.pub/2020/growing-ca/
