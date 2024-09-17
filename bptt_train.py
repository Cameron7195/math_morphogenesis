# Imports
from manim import *
from Organism_equiformer import *
import numpy as np
import torch
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 4             # Number of organisms to simulate in parallel
STATE_SIZE = 22            # 6 position & velocity, 16 cell-state features
LEARNING_RATE = 3e-4       # Learning rate
NUM_TRAIN_STEPS = 1000     # Number of trajectory rollout + backprop + gradient
                           # steps (basically num epochs)
TRAJECTORY_LENGTH = 520    # Number of timesteps to simulate for each organism
LOSS_TIMESTEPS = 150       # Number of timesteps to calculate loss over
D_MODEL = 32               # Hidden dimension of the equivariant transformer
N_HEADS = 4                # Number of attention heads
N_LAYERS = 1               # Number of transformer layers

device = torch.device("cpu")

f_nn = f_equiformer_net(STATE_SIZE,
                        d_model=D_MODEL,
                        n_heads=N_HEADS,
                        n_layers=N_LAYERS,
                        device=device)

# Print model
print(f_nn)

# print num params
num_params = sum(p.numel() for p in f_nn.parameters())
print(f"Number of parameters in model: {num_params}")

# Define optimizer. AdamW is current SOTA for transformers
optimizer = torch.optim.AdamW(f_nn.parameters(), lr=LEARNING_RATE)

best_loss = 100000
for train_step in range(NUM_TRAIN_STEPS):
    org = Organism(STATE_SIZE, f_nn, batch_size=BATCH_SIZE)
    optimizer.zero_grad()
    nonsphere_loss, neighbor_loss = (0.0, 0.0)
    for t in tqdm(range(0, TRAJECTORY_LENGTH)):
        org.evolve()
        
        # Calculate loss if we are in the last LOSS_TIMESTEPS timesteps
        if t > TRAJECTORY_LENGTH - LOSS_TIMESTEPS:
            nonsphere_penalty, neighbor_penalty = org.sphere_loss(org.X)

            nonsphere_loss += nonsphere_penalty.mean()
            neighbor_loss += neighbor_penalty.mean()
    
    # Calculate total loss for this trajectory
    traj_loss = (nonsphere_loss + neighbor_loss) / LOSS_TIMESTEPS

    print(f"Training step {train_step}, trajectory loss: {traj_loss.item()}")
    print(f"Avg Nonsphere loss: {nonsphere_loss.item()/LOSS_TIMESTEPS}")
    print(f"Avg Neighbor loss: {neighbor_loss.item()/LOSS_TIMESTEPS}")

    traj_loss.backward()
    optimizer.step()

    if traj_loss < best_loss:
        best_loss = traj_loss
        torch.save(f_nn.state_dict(), "./results/models/bptt_equiformer_model.pt")