
from manim import *
from Diffusion_equiformer import *
import numpy as np
import torch

# Note, these hyperparameters MUST match the ones used to train the model
STATE_SIZE = 22            # 6 position & velocity, 16 cell-state features
TRAJECTORY_LENGTH = 520    # Number of timesteps to simulate for each organism
D_MODEL = 24               # Hidden dimension of the equivariant transformer
N_HEADS = 4                # Number of attention heads
N_LAYERS = 1               # Number of transformer layers
OBJ_FILE = 'models/Jurassic_Saturnalid_Radiolarian.stl'

mesh = trimesh.load(OBJ_FILE)
mesh.apply_scale(3.0 / np.max(mesh.extents))
mesh.apply_translation(-mesh.centroid)
print("Loaded mesh. Number of vertices: ", len(mesh.vertices), "Number of faces: ", len(mesh.faces), "Mesh extents: ", mesh.extents)

device = torch.device("cpu")
f_nn = f_equiformer_net(STATE_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, device=device).to(device)
f_nn.eval()
# Load model

#f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model_rot_big.pt"))
#f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model_rot_2.pt"))


class GraphExample(Scene):
    def construct(self):
        pos = [np.array([0.0, 0.0, 0.0])]
        org = Organism(STATE_SIZE, f_nn)
        org.evolve_mesh_traj(mesh, m=512)

        V = org.get_vertices()
        E = org.get_edges()
        layout = org.get_layout()
        
        D = Graph(
            V,
            E,
            layout=layout,
            layout_scale=0.5
        )
        self.add(D)
        #self.play(D)
        self.wait()
        b = 8
        for i in range(0, 52):
            #D.vertices[1].move_to([1, 1 + i/100, 0])
            org.evolve_mesh_traj(mesh, m=512, blur_size=b)

            V = org.get_vertices()
            E = org.get_edges()
            layout = org.get_layout()

            self.remove(D)
            D = Graph(
                V,
                E,
                layout=layout,
                layout_scale=0.5
            )

            self.add(D)
            self.wait(0.1)
            b += 2


        self.wait()
        return