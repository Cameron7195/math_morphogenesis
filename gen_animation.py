
from manim import *
from Organism_equiformer import *
import numpy as np
import torch

# Note, these hyperparameters MUST match the ones used to train the model
STATE_SIZE = 22            # 6 position & velocity, 16 cell-state features
TRAJECTORY_LENGTH = 520    # Number of timesteps to simulate for each organism
D_MODEL = 24               # Hidden dimension of the equivariant transformer
N_HEADS = 4                # Number of attention heads
N_LAYERS = 1               # Number of transformer layers

device = torch.device("cpu")
f_nn = f_equiformer_net(STATE_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, device=device).to(device)
f_nn.eval()
# Load model

f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model_rot_big.pt"))
#f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model_rot_2.pt"))


class GraphExample(Scene):
    def construct(self):
        with torch.no_grad():
            pos = [np.array([0.0, 0.0, 0.0])]
            org = Organism(STATE_SIZE, f_nn)

            V = org.get_vertices()
            E = org.get_edges()
            colours = org.get_colours()
            layout = org.get_layout()
            
            D = Graph(
                V,
                E,
                vertex_config={
                    v: {"fill_color": rgb_to_color(colours[v][:3])} for v in V  # Applying the vertex colors
                },
                layout=layout
            )
            self.add(D)
            #self.play(D)
            self.wait()

            for i in range(0, 520):
                #D.vertices[1].move_to([1, 1 + i/100, 0])
                org.evolve()
                org.sphere_loss(org.X)

                V = org.get_vertices()
                E = org.get_edges()
                colours = org.get_colours()
                layout = org.get_layout()

                self.remove(D)
                D = Graph(
                    V,
                    E,
                    vertex_config={
                    v: {"fill_color": rgb_to_color(colours[v][:3])} for v in V  # Applying the vertex colors
                },
                    layout=layout
                )

                self.add(D)
                self.wait(0.1)
                print(org.X[:, :, 7])

            self.wait()
            return
    
def rgb_to_color(rgb):
    """Helper function to convert RGBA to a ManimColor object"""
    return ManimColor.from_rgb((rgb[0], rgb[1], rgb[2], 0.5))  # Use the ManimColor class with full opacity (alpha = 1)