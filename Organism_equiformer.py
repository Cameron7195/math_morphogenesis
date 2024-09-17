# Imports
import numpy as np
import torch
import torch.nn as nn
import copy
import pdb
import math
from equiformer_pytorch import Equiformer

class f_equiformer_net(torch.nn.Module):
    def __init__(self, n, d_model, n_heads, n_layers, device='cpu'):
        super(f_equiformer_net, self).__init__()
        self.n = n
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = device

        self.equiformer_model = Equiformer(dim=d_model,
                                           dim_in=n-6,
                                           dim_head=d_model//n_heads,
                                           heads=n_heads,
                                           radial_hidden_dim=d_model//8,
                                           num_degrees=2,
                                           depth=n_layers,
                                           num_neighbors=6,
                                           num_linear_attn_heads = 0)


        self.state_update_proj = nn.Linear(self.d_model, n-6)
        self.force_proj = nn.Linear(self.d_model, 1)

        init_params = torch.randn(1, n-6)
        init_params = init_params / (torch.norm(init_params, dim=1, keepdim=True) + 1e-8)
        self.init_state_params = torch.nn.Parameter(init_params, requires_grad=True)

    def forward(self, X):
        batch_size = X.shape[0]

        cell_positions = X[:, :, 0:3]  # [batch, m, 3]
        cell_velocities = X[:, :, 3:6] # Currently unused in computation. May be used in future.
        cell_states = X[:, :, 6:]      # [batch, m, n-6] 

        # Add a dummy position and feature to the input for m=1 (Equiformer requires m>1)
        extra_position = torch.zeros(batch_size, 1, 3).to(self.device)
        extra_state = torch.zeros(batch_size, 1, self.n-6).to(self.device)

        cell_positions_in = torch.cat((extra_position, cell_positions), dim=1)
        cell_states_in = torch.cat((extra_state, cell_states), dim=1)

        y = self.equiformer_model(cell_states_in, cell_positions_in)

        # Get the invariant output and remove the dummy position/feature
        y_inv = y.type0                 # [batch, m+1, d_model]
        y_inv = y_inv[:, 1:]            # [batch, m, d_model]

        # Get the equivariant output, remove dummy position/feature,
        # and put d_model last to project down.
        y_eq = y.type1                  # [batch, m+1, d_model, 3]
        y_eq = y_eq[:, 1:]              # [batch, m, d_model, 3]
        y_eq = y_eq.permute(0, 1, 3, 2) # [batch, m, 3, d_model]

        cell_state_update = self.state_update_proj(y_inv)
        cell_forces = self.force_proj(y_eq).squeeze(-1)

        # Enforce F_net=0 for the entire organism.
        net_force = cell_forces.mean(dim=1, keepdim=True)
        cell_forces = cell_forces - net_force

        return cell_state_update, cell_forces

class Organism():
    def __init__(self,
                 n,
                 f_nn,
                 batch_size=1,
                 noise=0.04,
                 dt=0.1,
                 friction_coeff=4.0):
        if n < 6:
            raise ValueError("Cell State must have at least 6 dimensions (position and velocity).")

        self.f_nn = f_nn
        self.device = f_nn.device
        self.batch_size = batch_size
        self.noise = noise

        init_pos_vel = torch.zeros(batch_size, 1, 6).to(self.device)
        self.X = torch.cat((init_pos_vel, self.f_nn.init_state_params.repeat(batch_size, 1, 1)), dim=-1)

        # Add noise to initial state, excluding position and velocity
        self.X[:, :, 6:] += self.noise*torch.randn_like(self.X[:, :, 6:])

        self.n = n
        self.m = 1
        self.t = 0.0
        self.b = friction_coeff
        self.dt = dt

    def evolve(self):
        # Euler method
        new_X = self.X + self.dt*self.f(self.X)

        # Take the normalized direction of the cell state. We artificially enforce
        # the cell state to be a unit vector, to ensure numerical stability during
        # training.
        x_h_norm = new_X[:, :, 6:] / (torch.norm(new_X[:, :, 6:], dim=2, keepdim=True) + 1e-8)

        # Piece together the new X.
        self.X = torch.cat((new_X[:, :, 0:6], x_h_norm), dim=2)

        # Call d(x) to kill any cells that should die. Currently unimplemented.
        self.die(self.X)

        # Call b(x) to divide any cells that should divide.
        self.divide(self.X)

        self.t += 1
        return
    
    def f(self, X):
        assert not torch.any(torch.isnan(X)) # Safety check!

        X_prime = torch.zeros((self.batch_size, self.m, self.n)).to(self.device)

        # dx/dt = v
        X_prime[:, :, 0:3] = X[:, :, 3:6]

        cell_state_update, cell_force = self.f_nn(X) # [batch, m, n-6], [batch, m, 3]

        # Add noise to cell state update and cell force
        cell_state_update += self.noise*torch.randn_like(cell_state_update)
        cell_force += self.noise*torch.randn_like(cell_force)

        cell_velocities = X[:, :, 3:6]
        friction_force = -self.b*cell_velocities

        X_prime[:, :, 3:6] = cell_force + friction_force
        X_prime[:, :, 6:] = cell_state_update

        return X_prime
    
    def divide(self, X):
        # Cell division function. b(x) = {0, 1}
        # Currently, every cell divides every 80 timesteps.
        # TODO: Implement a 'mass' mechanism (effectively measuring mitogenic 'progress')
        # which will determine when a cell divides, and play a role in loss function
        # so that b(x) can depend on a neural network and be learned.
        
        if self.m < 32 and self.t % 80 == 1:
            new_xi = X + 0.1*self.noise*torch.randn_like(X).to(self.device) # [batch, m, n]
            self.X = torch.cat((self.X, new_xi), dim=1)                     # [batch, 2m, n]
            self.m *= 2

        return
    
    def die(self, X):
        # Placeholder for cell death. d(x) = {0, 1}. Currently unimplemented.
        return
    
    def sphere_loss(self,
                    X,
                    R=3.0,
                    nonsphere_coeff=1.0,
                    neighbor_coeff=0.05,
                    beat_heart=True,
                    num_neighbors=3,
                    desired_neighbor_dist=1.2):
        
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)  # [batch, m, m]

        desired_r = R + self.heartbeat_signal(self.t) if beat_heart else R

        nonsphere_penalty = ((cell_positions.norm(dim=-1) - desired_r)**2).mean(dim=-1)

        # Mask out the diagonal and only consider the (num_neighbors) closest neighbors
        neighbor_mask = cell_distances.argsort(dim=2).argsort(dim=2) < num_neighbors+1
        diag_mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)

        neighbor_penalty = (desired_neighbor_dist - cell_distances)**2 * neighbor_mask
        neighbor_penalty[diag_mask] = 0.0

        neighbor_penalty = torch.sum(neighbor_penalty, dim=(1, 2))

        return nonsphere_coeff * nonsphere_penalty, neighbor_coeff * neighbor_penalty
    
    def heartbeat_signal(self, t, amplitude=0.5, period=25):
        # Returns a heartbeat-like signal, given a timestep
        # Adapted from here: https://www.desmos.com/calculator/2bqvtdd6vd
        return amplitude * 0.3116*(
                (math.sin((9.4248/period)*t * 4) + (math.sin((9.4248/period)*t * 16) / 4)) * 3 *
                (-(math.floor(math.sin((9.4248/period)*t * 2)) + 0.1)) *
                (1 - math.floor(math.fmod(math.sin((9.4248/period)*t / 1.5), 2))))
    
    # These functions are for manim (animation software used for visualizing the simulation)
    def get_vertices(self):
        return [v for v in range(self.X.shape[1])]
    
    def get_edges(self, connected_threshold=2.0):
        vertex_positions = self.X[0, :, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2).detach().numpy()

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if vertex_distances[i, j] < connected_threshold:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[0, i, 0:3].detach().numpy() for i in range(self.m)}