# Imports
#from manim import *
from Organism_equiformer import *
import numpy as np
import torch
import trimesh
from tqdm import tqdm
import os

# Hyperparameters
BATCH_SIZE = 2             # Number of organisms to simulate in parallel
STATE_SIZE = 14            # 6 position & velocity, 16 cell-state features
LEARNING_RATE = 1e-3       # Learning rate
NUM_TRAIN_STEPS = 10000    # Number of trajectory rollout + backprop + gradient
                           # steps (basically num epochs)
TRAJECTORY_LENGTH = 520    # Number of timesteps to simulate for each organism
LOSS_TIMESTEPS = 300       # Number of timesteps to calculate loss over
D_MODEL = 24               # Hidden dimension of the equivariant transformer
N_HEADS = 4                # Number of attention heads
N_LAYERS = 1               # Number of transformer layers
OBJ_FILE = 'models/hand_small.obj'

device = torch.device("cpu")

f_nn = f_equiformer_net(STATE_SIZE,
                        d_model=D_MODEL,
                        n_heads=N_HEADS,
                        n_layers=N_LAYERS,
                        device=device)
#f_nn.load_state_dict(torch.load("results/models/bptt_equiformer_model1.pt"))

# Load the mesh
mesh = trimesh.load(OBJ_FILE)
mesh.apply_scale(3.0 / np.max(mesh.extents))
mesh.apply_translation(-mesh.centroid)
print("Loaded mesh. Number of vertices: ", len(mesh.vertices), "Number of faces: ", len(mesh.faces), "Mesh extents: ", mesh.extents)
# TODO figure out how to make this float32

torch.manual_seed(0)
# Print model
print(f_nn)

# print num params
num_params = sum(p.numel() for p in f_nn.parameters())
print(f"Number of parameters in model: {num_params}")

# Define optimizer. AdamW is current SOTA for transformers
optimizer = torch.optim.AdamW(f_nn.parameters(), lr=LEARNING_RATE)

best_loss = 100000
for train_step in range(NUM_TRAIN_STEPS):
    f_nn.s4_inputs = None
    f_nn.s4_outputs = None
    org = Organism(STATE_SIZE, f_nn, batch_size=BATCH_SIZE, noise=0.01)
    optimizer.zero_grad()
    nonsphere_loss, neighbor_loss, corr_loss, ortho_chiral_loss = (0.0, 0.0, 0.0, 0.0)
    corr_loss1_total, corr_loss2_total, corr_loss3_total, corr_loss_radial_total = (0.0, 0.0, 0.0, 0.0)
    w_norm_loss = 0.0

    for t in tqdm(range(0, TRAJECTORY_LENGTH)):
        org.evolve()
        
        # Calculate loss if we are in the last LOSS_TIMESTEPS timesteps
        if t > TRAJECTORY_LENGTH - LOSS_TIMESTEPS:
            #nonsphere_penalty, neighbor_penalty = org.sphere_loss(org.X)

            #if t > 270:
            #    nonsphere_loss += org.sphere_loss(org.X).mean()
            corr_loss1, w1 = org.compute_vector_scalar_correlation(org.X, morphogen_idx=6)
            corr_loss2, w2 = org.compute_vector_scalar_correlation(org.X, morphogen_idx=7)
            #corr_loss3, w3 = org.compute_vector_scalar_correlation(org.X, morphogen_idx=8, project1=w1, project2=w2)

            corr_loss_radial, _ = org.compute_vector_scalar_correlation(org.X, morphogen_idx=9, radial=True)


            # m6m7_corr_loss = 1.0*org.compute_vector_scalar_correlation(org.X, morphogen_idx=7, override_w=w1)[0]
            # m6m8_corr_loss = 0.5*org.compute_vector_scalar_correlation(org.X, morphogen_idx=8, override_w=w1)[0]
            # m7m8_corr_loss = 0.5*org.compute_vector_scalar_correlation(org.X, morphogen_idx=8, override_w=w2)[0]

            w_norm_loss -= 0*(w1.norm(dim=-1).sqrt().mean() + w2.norm(dim=-1).sqrt().mean() )

            m6m7_corr_loss = org.compute_morphogen_correlation(org.X, morphogen_idx1=6, morphogen_idx2=7)
            #m6m8_corr_loss = org.compute_morphogen_correlation(org.X, morphogen_idx1=6, morphogen_idx2=8)
            #m7m8_corr_loss = org.compute_morphogen_correlation(org.X, morphogen_idx1=7, morphogen_idx2=8)


            ortho_chiral_loss += 3*(m6m7_corr_loss.mean())
            
            corr_loss1_total -= corr_loss1.mean()*1
            corr_loss2_total -= corr_loss2.mean()*1
            corr_loss3_total -= 0#corr_loss3.mean()*1
            #corr_loss_radial_total -= 0.25*corr_loss_radial.mean()
            #ortho_chiral_loss = 0.0
            #corr_loss -= org.compute_total_morphogen_correlation(org.X).mean()
            neighbor_loss += 2*org.neighbour_loss(org.X).mean()

            #nonsphere_loss += org.obj_loss(org.X, mesh).mean()

    # Calculate total loss for this trajectory
    traj_loss = (nonsphere_loss + neighbor_loss + corr_loss1_total + corr_loss2_total + corr_loss3_total + corr_loss_radial_total + ortho_chiral_loss + w_norm_loss) / LOSS_TIMESTEPS

    vel_loss = 0.001*org.squared_vels.mean()
    state_vel_loss = 0.0001*org.squared_state_vels.mean()

    #traj_loss += vel_loss + state_vel_loss

    #print(org.f_nn.init_state_params.norm(dim=-1).mean())

    print(f"Training step {train_step}, trajectory loss: {traj_loss.item()}")
    #print(f"Avg Nonsphere loss: {nonsphere_loss.item()/LOSS_TIMESTEPS}")
    print(f"Avg Correlation loss 1: {corr_loss1_total.item()/LOSS_TIMESTEPS} / {-20}")
    print(f"Avg Correlation loss 2: {corr_loss2_total.item()/LOSS_TIMESTEPS} / {-20}")
    
    #print(f"Avg Correlation loss 3: {corr_loss3_total.item()/LOSS_TIMESTEPS}")
    #print(f"Avg Correlation loss radial: {corr_loss_radial_total.item()/LOSS_TIMESTEPS}")
    print(f"Avg w norm loss: {w_norm_loss.item()/LOSS_TIMESTEPS}")
    print(f"Avg Neighbor loss: {neighbor_loss.item()/LOSS_TIMESTEPS}")
    print(f"Avg vel loss:  {vel_loss.item()}")
    print(f"Avg ortho chiral loss: {ortho_chiral_loss.item()/LOSS_TIMESTEPS} / {40}")
    print(f"Avg state vel loss: {state_vel_loss.item()}")

    traj_loss.backward()
    optimizer.step()

    if traj_loss < best_loss:
        best_loss = traj_loss
        os.makedirs("results/models", exist_ok=True)
        torch.save(f_nn.state_dict(), "results/models/bptt_grad_normal_1.pt")