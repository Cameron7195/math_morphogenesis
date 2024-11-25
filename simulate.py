import argparse
from Organism_equiformer import *
import torch
from tqdm import tqdm
import trimesh

STATE_SIZE = 14            # 6 position & velocity, 16 cell-state features
D_MODEL = 24               # Hidden dimension of the equivariant transformer
N_HEADS = 4                # Number of attention heads
N_LAYERS = 1               # Number of transformer layers
OBJ_FILE = 'models/hand_small.obj'

device = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='S4 Model')
    parser.add_argument('--num_timesteps', type=int, default=520, help='Number of timesteps to simulate for each organism')
    parser.add_argument('--model_path', type=str, default='results/models/bptt_grad_normal_1_osc1.pt', help='Path to model')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    print(args)
    print(args.num_timesteps)
    print(args.model_path)
    print('Hello, World!')

    # load model
    f_nn = f_equiformer_net(STATE_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, device=device).to(device)

    f_nn.load_state_dict(torch.load(args.model_path))

    print(f_nn)
    print('Number of parameters in model: ', sum(p.numel() for p in f_nn.parameters()))

    # Load the mesh
    mesh = trimesh.load(OBJ_FILE)
    mesh.apply_scale(3.0 / np.max(mesh.extents))
    mesh.apply_translation(-mesh.centroid)
    print("Loaded mesh. Number of vertices: ", len(mesh.vertices), "Number of faces: ", len(mesh.faces), "Mesh extents: ", mesh.extents)

    # simulate
    #org = Organism(STATE_SIZE, f_nn, obj_out_dir='results/obj_out_1', mesh_save=mesh)
    org = Organism(STATE_SIZE, f_nn, obj_out_dir='results/obj_out_1')

    for t in tqdm(range(0, args.num_timesteps)):
        org.evolve()