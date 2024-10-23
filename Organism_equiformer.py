# Imports
import numpy as np
import torch
import torch.nn as nn
import copy
import pdb
import math
from equiformer_pytorch import Equiformer
from s4torch import S4Model
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import trimesh
import os
from chamferdist import ChamferDistance
from torch.autograd import grad
from torch.autograd.functional import hessian
from scipy.optimize import differential_evolution
from egnn_pytorch import EGNN_Network
import matplotlib.pyplot as plt

# import PCA
from sklearn.decomposition import PCA


import torch
import numpy as np
from scipy.special import sph_harm, gammaln

import torch

def associated_legendre_polynomials(l, m, x):
    """
    Computes the associated Legendre polynomials P_l^m(x)
    """
    m_abs = abs(m)
    # Initialize P_m^m
    P_mm = (-1)**m_abs * torch.pow(1 - x**2 + 1e-8, m_abs / 2)
    if l == m_abs:
        return P_mm
    # Compute P_{m+1}^m
    P_m1m = x * (2 * m_abs + 1) * P_mm
    if l == m_abs + 1:
        return P_m1m
    # Recurrence for l > m + 1
    P_lm_prev = P_mm
    P_lm = P_m1m
    for ll in range(m_abs + 2, l + 1):
        P_lm_new = ((2 * ll - 1) * x * P_lm - (ll + m_abs - 1) * P_lm_prev) / (ll - m_abs)
        P_lm_prev = P_lm
        P_lm = P_lm_new
    return P_lm

def spherical_harmonics(l, m, theta, phi):
    """
    Computes the spherical harmonics Y_l^m(theta, phi)
    """
    m_abs = abs(m)
    P_lm = associated_legendre_polynomials(l, m_abs, torch.cos(theta))  # (...), real
    normalization = torch.sqrt((2 * l + 1) / (4 * torch.pi) * torch.exp(
        torch.lgamma(torch.tensor(l - m_abs + 1.0)) - torch.lgamma(torch.tensor(l + m_abs + 1.0))
    ))
    if m >= 0:
        Y_lm = normalization * P_lm * torch.exp(1j * m * phi)
    else:
        # Use the Condon-Shortley phase for negative m
        Y_lm = normalization * P_lm * torch.exp(1j * m * phi) * (-1)**m_abs
    return Y_lm  # (...), complex

def compute_radial_coefficients(n, l):
    """
    Computes the radial coefficients c_nlk for given n and l
    """
    k_max = (n - l) // 2
    c_nlk = []
    for k in range(k_max + 1):
        num = (-1)**k * torch.exp(
            torch.lgamma(torch.tensor(n - k + 1.0)) - 
            (torch.lgamma(torch.tensor(k + 1.0)) + 
             torch.lgamma(torch.tensor((n + l) / 2 - k + 1.0)) + 
             torch.lgamma(torch.tensor((n - l) / 2 - k + 1.0)))
        )
        c_nlk.append(num)
    return c_nlk  # list of tensors

def compute_radial_polynomial(r_scaled, n, l, c_nlk):
    """
    Computes the radial polynomial R_nl(r_scaled)
    """
    k_max = (n - l) // 2
    R_nl = torch.zeros_like(r_scaled)
    for k in range(k_max + 1):
        c = c_nlk[k]
        exponent = 2 * k + l
        R_nl += c * r_scaled.pow(exponent)
    return R_nl

def compute_zernike_moments(point_clouds, N_max):
    """
    Computes the Zernike moments of a batched 3D point cloud, scaling to the unit sphere.
    
    Args:
        point_clouds: torch tensor of shape (B, N, 3)
        N_max: maximum order of Zernike polynomials
    
    Returns:
        moments: torch tensor of shape (B, num_moments), complex dtype
        scales: torch tensor of shape (B,), the scaling factors used
    """
    B, N, _ = point_clouds.shape
    x = point_clouds  # (B, N, 3)
    # Center the point clouds
    x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, 3)
    x_centered = x - x_mean  # Centered point clouds
    # Compute the maximum radius R_max for each point cloud
    r = torch.linalg.norm(x_centered, dim=2)  # (B, N)
    R_max = r.max(dim=1, keepdim=True)[0]  # (B, 1)
    scales = r.mean(dim=1)
    scale_max = R_max.squeeze(-1)
    # Avoid division by zero
    R_max_safe = R_max + 1e-8
    # Scale the point clouds to the unit sphere
    x_scaled = x_centered / R_max_safe  # (B, N, 3)
    r_scaled = r / R_max_safe  # (B, N)
    # Compute theta and phi
    theta = torch.acos(torch.clamp(x_scaled[:, :, 2] / (r_scaled + 1e-8), -1.0, 1.0))  # (B, N)
    phi = torch.atan2(x_scaled[:, :, 1], x_scaled[:, :, 0])  # (B, N)
    # Handle NaNs
    theta = torch.nan_to_num(theta, nan=0.0)
    phi = torch.nan_to_num(phi, nan=0.0)
    # Generate indices (n, l, m)
    indices = []
    for n in range(N_max + 1):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                for m in range(-l, l + 1):
                    indices.append((n, l, m))
    num_moments = len(indices)
    moments = torch.zeros((B, num_moments), dtype=torch.complex64, device=x.device)
    # Precompute c_nlk coefficients for each (n,l)
    c_nl_dict = {}
    for n in range(N_max + 1):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                c_nlk = compute_radial_coefficients(n, l)
                c_nl_dict[(n, l)] = c_nlk
    # For each (n,l,m), compute moments
    for idx, (n, l, m) in enumerate(indices):
        c_nlk = c_nl_dict[(n, l)]
        # Compute R_nl(r_scaled) of shape (B, N)
        R_nl = compute_radial_polynomial(r_scaled, n, l, c_nlk)  # (B, N)
        # Compute Y_lm(θ, φ) of shape (B, N)
        Y_lm = spherical_harmonics(l, m, theta, phi)  # (B, N), complex
        # Compute Z_nlm(x_i) = R_nl(r_i) * Y_lm(θ_i, φ_i), shape (B, N)
        Z_nlm = R_nl * Y_lm  # (B, N), complex
        # Compute Ω_nlm = sum_i Z_nlm[i] * w_i
        Omega_nlm = Z_nlm.mean(dim=1)  # (B,)
        moments[:, idx] = Omega_nlm  # Store in moments array
    return moments, scales, scale_max  # (B, num_moments), complex tensor and (B,) scaling factors

class f_equiformer_net(torch.nn.Module):
    def __init__(self, n, d_model, n_heads, n_layers, device='cpu'):
        super(f_equiformer_net, self).__init__()
        self.n = n
        self.input_dim = n-6#(n-6)//2
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = device

        self.equiformer_model = Equiformer(dim=d_model,
                                           dim_in=self.input_dim,
                                           dim_head=d_model//n_heads,
                                           heads=n_heads,
                                           radial_hidden_dim=3*d_model//n_heads,
                                           num_degrees=2,
                                           depth=n_layers,
                                           num_neighbors=6,
                                           num_linear_attn_heads = 1)
        
        # self.egnn_net = EGNN_Network(depth=n_layers,
        #                 dim=self.input_dim,
        #                 m_dim = d_model,
        #                 num_nearest_neighbors=6,        # number of nearest neighbors to consider
        #                 global_linear_attn_heads=0,
        #                 global_linear_attn_every=0,
        #                 num_global_tokens=0,
        #                 valid_radius=1e4,
        #                 fourier_features = 16)
        

        self.state_update_proj = nn.Linear(self.d_model, self.d_model)
        self.num_k = 512
        self.largest_period = 1000
        #self.fourier_uproj = nn.Linear(d_model, 2*self.num_k)
        #self.fourier_dproj = nn.Linear(2*self.num_k, self.d_model)
        self.force_proj = nn.Linear(self.d_model, 1)
        self.act = nn.GELU()
        self.desire_dproj = nn.Linear(self.d_model, self.input_dim)
        self.basis_matrices = 1
        self.desire_basis_proj = nn.Linear(self.d_model, self.basis_matrices*self.input_dim)

        # self.s4_model = S4Model(d_input=self.input_dim+3,
        #                         d_model=self.d_model,
        #                         d_output=self.input_dim,
        #                         n_blocks=1,
        #                         n=8,
        #                         l_max=520,)
        
        # self.s4_inputs = None
        # self.s4_outputs = None

        init_params = torch.randn(1, self.input_dim) / 2.7
        #init_params = init_params / (torch.norm(init_params, dim=1, keepdim=True) + 1e-8)
        self.init_state_params = torch.nn.Parameter(init_params, requires_grad=True)

        #self.fourier_coeffs = torch.nn.Parameter(torch.randn(1, 2*self.num_k), requires_grad=True)
        #self.coefficient_factor = torch.arange(1, self.num_k+1, 1).float().to(self.device)
        #self.coefficient_factor = self.coefficient_factor / (self.coefficient_factor + 1e-8)

        #self.rot_oproj_params = torch.nn.Parameter(1/self.basis_matrices + torch.randn(self.basis_matrices)/math.sqrt(self.basis_matrices), requires_grad=True)

        self.dt = 0.1

    def forward(self, X, t):
        batch_size = X.shape[0]
        m = X.shape[1]

        cell_positions = X[:, :, 0:3]  # [batch, m, 3]
        cell_velocities = X[:, :, 3:6] # Currently unused in computation. May be used in future.
        
        cell_states = X[:, :, 6:]
        #cell_states = X[:, :, 6:6+self.input_dim]      # [batch, m, n-6] 
        #cell_states_velocities = X[:, :, 6+self.input_dim:] # Currently unused in computation. May be used in future.
        # if self.s4_outputs == None:
        #     cell_states = torch.zeros(batch_size, m, self.input_dim).to(self.device)
        # else:
        #     if self.s4_outputs.shape[0] // batch_size != m:
        #         # We divided last timestep. Need to repeat batch dim to match batch * m.
        #         self.s4_outputs = self.s4_outputs.repeat(2, 1, 1)
        #     cell_states = self.s4_outputs[:, -1].view(batch_size, m, self.input_dim)

        # Add a dummy position and feature to the input for m=1 (Equiformer requires m>1)
        extra_position = torch.zeros(batch_size, 1, 3).to(self.device)
        extra_state = torch.zeros(batch_size, 1, self.input_dim).to(self.device)

        cell_positions_in = torch.cat((extra_position, cell_positions), dim=1)
        cell_states_in = torch.cat((extra_state, cell_states), dim=1)

        y = self.equiformer_model(cell_states_in, cell_positions_in)

        # Get the invariant output and remove the dummy position/feature
        y_inv = y.type0                 # [batch, m+1, d_model]
        #y_inv = y[0]
        y_inv = y_inv[:, 1:]            # [batch, m, d_model]

        # Get the equivariant output, remove dummy position/feature,
        # and put d_model last to project down.
        
        y_eq = y.type1                  # [batch, m+1, d_model, 3]
        #y_eq = y[1]
        y_eq = y_eq[:, 1:]              # [batch, m, d_model, 3]
        y_eq = y_eq.permute(0, 1, 3, 2) # [batch, m, 3, d_model]

        cell_state_desire = self.desire_dproj(y_inv)

        cell_desires = self.force_proj(y_eq).squeeze(-1)
        cell_forces = cell_desires - cell_positions

        # if self.s4_inputs is None:
        #     y_inv_reshape = cell_state_desire.view(batch_size*m, 1, self.input_dim)
        #     self.s4_inputs = y_inv_reshape
        #     # cat with cell_desires
        #     cd_reshape = cell_desires.view(batch_size*m, 1, 3)
        #     self.s4_inputs = torch.cat((self.s4_inputs, cd_reshape), dim=-1)
        # else:
        #     if self.s4_inputs.shape[0] // batch_size != m:
        #         # We divided last timestep. Need to repeat batch dim to match batch * m.
        #         self.s4_inputs = self.s4_inputs.repeat(2, 1, 1)
        #     #pdb.set_trace()
        #     y_inv_reshape = cell_state_desire.view(batch_size*m, 1, self.input_dim)
        #     # cat with cell_desires
        #     cd_reshape = cell_desires.view(batch_size*m, 1, 3)
        #     y_inv_reshape = torch.cat((y_inv_reshape, cd_reshape), dim=-1)

        #     self.s4_inputs = torch.cat((self.s4_inputs, y_inv_reshape), dim=1)

        # if self.s4_inputs.shape[1] < 520:
        #     # We need to pad the input to 520 timesteps
        #     pad_amount = 520 - self.s4_inputs.shape[1]
        #     pad = torch.zeros(batch_size*m, pad_amount, self.input_dim+3).to(self.device)
        #     pad_inps = torch.cat((self.s4_inputs, pad), dim=1)
        # else:
        #     pad_amount = 0
        #     pad_inps = self.s4_inputs[:, -520:]
        
        # s4_output = self.s4_model(pad_inps)
        # self.s4_outputs = s4_output[:, :-pad_amount]

        cell_state_urges = cell_state_desire - cell_states
        net_urge = cell_state_urges.mean(dim=1, keepdim=True)
        cell_state_urges = cell_state_urges - net_urge


        # Enforce F_net=0 for the entire organism.
        net_force = cell_forces.mean(dim=1, keepdim=True)
        cell_forces = cell_forces - net_force
        centroid = cell_positions.mean(dim=1, keepdim=True)
        #net_torque = torch.cross(cell_positions - centroid, cell_forces).sum(dim=1, keepdim=True)
        #force_adjustment = torch.cross(cell_positions - centroid, net_torque)
        #cell_forces = cell_forces - force_adjustment
        
        # new_cell_states = torch.cos(theta)*cell_states[:, :, None, :] + torch.sin(theta) * e2
        
        # w = self.rot_oproj_params / (torch.sum(self.rot_oproj_params, dim=0, keepdim=True) + 1e-8)
        # new_cell_states = torch.einsum('ijkl, k -> ijl', new_cell_states, w)

        return cell_state_urges, cell_forces

class Organism():
    def __init__(self,
                 n,
                 f_nn,
                 batch_size=1,
                 noise=0.01,
                 dt=0.1,
                 friction_coeff=4.0,
                 obj_out_dir=None,
                 mesh_save=None):
        if n < 6:
            raise ValueError("Cell State must have at least 6 dimensions (position and velocity).")

        self.f_nn = f_nn
        self.device = f_nn.device
        self.batch_size = batch_size
        self.noise = noise

        init_pos_vel = torch.zeros(batch_size, 1, 6).to(self.device)
        init_state_params = self.f_nn.init_state_params.repeat(batch_size, 1, 1)
        #init_state_params = init_state_params / (torch.norm(init_state_params, dim=2, keepdim=True) + 1e-8)
        init_state_vel = torch.zeros_like(init_state_params)
        #self.X = torch.cat((init_pos_vel, init_state_params, init_state_vel), dim=-1)
        self.X = torch.cat((init_pos_vel, init_state_params), dim=-1) 
        #self.X = torch.zeros(batch_size, 1, n).to(self.device)

        # Add noise to initial state, excluding position and velocity
        #self.X[:, :, 6:] += self.noise*torch.randn_like(self.X[:, :, 6:])

        self.n = n
        self.m = 1
        self.t = 0
        self.b = friction_coeff
        self.dt = dt

        self.init_guess_params = torch.zeros(batch_size, 4).to(self.device)
        self.init_guess_params[:, 0] = 1.0

        self.obj_out_dir = obj_out_dir
        if mesh_save is not None:
            self.mesh_save = mesh_save
        else:
            self.mesh_save = None
        if self.obj_out_dir is not None:
            #if os.path.exists(self.obj_out_dir):
            #    os.system(f'rm -r {self.obj_out_dir}')
            os.makedirs(self.obj_out_dir, exist_ok=True)
            self.write_obj(self.mesh_save)


    def evolve(self):
        # Euler method
        fx = self.f(self.X)
        self.X = self.X + self.dt*fx

        # x_h_norm = fx[:, :, 6:] / (torch.norm(fx[:, :, 6:], dim=2, keepdim=True))
        # #rotation_matrix = self.rotary_encoding_matrix(self.n-6, device=self.device)
        # #x_h_norm = torch.einsum('ijk, kl -> ijl', self.X[:, :, 6:], rotation_matrix)
        # # Piece together the new X.
        # #self.X = torch.cat((new_X[:, :, 0:6], x_h_norm), dim=2)

        # self.X = new_X
        # Call d(x) to kill any cells that should die. Currently unimplemented.
        self.die(self.X)

        # Call b(x) to divide any cells that should divide.
        self.divide(self.X)

        self.t += 1
        if self.obj_out_dir is not None:
            self.write_obj(self.mesh_save)
        return
    
    def f(self, X):
        assert not torch.any(torch.isnan(X)) # Safety check!

        X_prime = torch.zeros((self.batch_size, self.m, self.n)).to(self.device)

        # dx/dt = v
        X_prime[:, :, 0:3] = X[:, :, 3:6]
        #X_prime[:, :, 6:6+(self.n-6)//2] = X[:, :, 6+(self.n-6)//2:]

        cell_state_update, cell_force = self.f_nn(X, self.t) # [batch, m, n-6], [batch, m, 3]

        # Add noise to cell state update and cell force
        cell_state_update += self.noise*torch.randn_like(cell_state_update)
        cell_force += self.noise*torch.randn_like(cell_force)

        cell_velocities = X[:, :, 3:6]
        #state_velocities = X[:, :, 6+(self.n-6)//2:]
        
        friction_force = -self.b*cell_velocities
        #friction_state_force = -self.b*state_velocities

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
            new_xi = X
            if self.m <= 2:
                new_xi = new_xi + 0.1*self.noise*torch.ones_like(X).to(self.device) # [batch, 1, n]
            else:
                new_xi = new_xi + 0.1*self.noise*torch.randn_like(X).to(self.device) # [batch, m, n]
            self.X = torch.cat((self.X, new_xi), dim=1)                  # [batch, 2m, n]
            self.m *= 2

        return
    
    def die(self, X):
        # Placeholder for cell death. d(x) = {0, 1}. Currently unimplemented.
        return
    
    def rotary_encoding_matrix(self, state_size, device='cpu'):
        # Returns a rotary encoding matrix for the given dimensions
        # Adapted from: https://arxiv.org/abs/2104.09864
        # and

        # Create the rotary encoding matrix
        num_freqs = state_size // 2
        min_period = 2
        max_period = 1000

        # Logarithmically spaced frequencies
        freqs = torch.logspace(math.log2(min_period), math.log2(max_period), num_freqs-1, base=2.0).to(device)

        # Append a 0 freq
        freqs = torch.cat((torch.zeros(1).to(device), freqs))

        # Create the rotation matrix
        encoding_matrix = torch.zeros(state_size, state_size).to(device)
        for i in range(num_freqs):
            angle = 2 * math.pi * freqs[i] / max_period
            encoding_matrix[2*i, 2*i] = math.cos(angle)
            encoding_matrix[2*i, 2*i+1] = -math.sin(angle)
            encoding_matrix[2*i+1, 2*i] = math.sin(angle)
            encoding_matrix[2*i+1, 2*i+1] = math.cos(angle)

        return encoding_matrix

    def obj_loss(self, X, mesh):

        def compute_jacobian(params, point_cloud, mesh, optimizing=False):
            """
            Compute the Jacobian of the error function with respect to the quaternion parameters.
            """
            # Ensure params require gradient computation
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True).to(self.device)  # Assuming you're using GPU

            # Call the error function
            loss = compute_error(params, point_cloud, mesh, optimizing=False)

            # Compute gradients (Jacobian)
            jacobians = []
            for i in range(loss.shape[0]):  # Iterate over the batch
                # Compute the gradient of the loss with respect to params[i]
                gradient = grad(loss[i], params, create_graph=True)[0]
                jacobians.append(gradient.cpu().numpy())  # Move to CPU if needed

            # Convert list of jacobians into an array
            return np.array(jacobians)
        
        def compute_hessian(params, point_cloud, mesh, optimizing=False):
            """
            Compute the Hessian of the error function with respect to the quaternion parameters.
            """
            # Ensure params require gradient computation
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True).to(self.device)  # Assuming you're using GPU

            # Define a wrapper for the function to compute the Hessian
            def wrapped_error(p):
                p_tensor = torch.tensor(p, dtype=torch.float32, requires_grad=True).to(self.device)
                return compute_error(p_tensor, point_cloud, mesh, optimizing=True)

            # Compute the Hessian
            hess = hessian(wrapped_error, params)
            return hess.cpu().numpy()  # Move to CPU if necessary and return as numpy array
        
        def compute_error(params, point_cloud, mesh, optimizing=False):
            """
            Compute the sum of squared distances from the transformed point cloud to the mesh surface.
            The transformation includes rotation and translation parameters.
            """
            pc1 = point_cloud
            pc2 = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)[None, :, :]
            # Take random sampling of 1% of the mesh vertices
            idx = torch.randperm(pc2.shape[1])[:int(0.01*pc2.shape[1])]
            pc2 = pc2[:, idx, :]
            pc2 = pc2.repeat(pc1.shape[0], 1, 1)

            # First, we'll centre the point cloud
            pc1 = pc1 - pc1.mean(dim=1, keepdim=True)
            pc2 = pc2 - pc2.mean(dim=1, keepdim=True)

            # Next, we'll rotate the point cloud to align as best as possible with the mesh
            # cov_pc1 = torch.bmm(pc1.transpose(1, 2), pc1) / (pc1.shape[1] - 1)  # Shape: (batch_size, 3, 3)
            # cov_pc2 = torch.bmm(pc2.transpose(1, 2), pc2) / (pc2.shape[1] - 1)  # Shape: (1, 3, 3)
            # e_vecs1 = torch.linalg.eigh(cov_pc1).eigenvectors.flip(dims=[2])     # Flip for descending order
            # e_vecs1 = torch.cat([e_vecs1[:, 0:2, :], torch.cross(e_vecs1[:, 0, :], e_vecs1[:, 1, :], dim=-1)[:, None, :]], dim=1)       # Ensure right-handed coordinate system
            # e_vecs2 = torch.linalg.eigh(cov_pc2).eigenvectors.flip(dims=[2])
            # e_vecs2 = torch.cat([e_vecs2[:, 0:2, :], torch.cross(e_vecs2[:, 0, :], e_vecs2[:, 1, :], dim=-1)[:, None, :]], dim=1)       # Ensure right-handed coordinate system
            # rot_mats = torch.bmm(e_vecs2, e_vecs1.transpose(1, 2))  # Shape: (batch_size, 3, 3)
            # pc1 = torch.bmm(pc1, rot_mats.transpose(1, 2))  # Apply rotation to pc1

            # params is shape (batch_size, 3)
            #rot_mats = torch.tensor(R.from_quat(params.reshape(self.batch_size, 4)).as_matrix(), dtype=torch.float).to(self.device)
            #pdb.set_trace()
            #pc1 = torch.bmm(pc1, rot_mats.transpose(1, 2))  # Apply rotation to pc1
            
            chamdist = ChamferDistance()
            chamfer_loss = chamdist(pc1, pc2, bidirectional=True, point_reduction='mean', batch_reduction=None)

            # dists = torch.cdist(pc1, pc2, p=2)


            # temp = 0.1
            # neg_log_dists = -torch.log(dists)
            # pc1_attention = torch.softmax(neg_log_dists / temp, dim=1)
            # pc2_attention = torch.softmax(neg_log_dists / temp, dim=2)

            # chamfer_loss = torch.sum(pc1_attention * dists**2, dim=1).mean(dim=-1) + torch.sum(pc2_attention * dists, dim=2).mean(dim=-1)

            # Return the sum of squared distances
            if optimizing:
                return chamfer_loss.mean()
            else:
                return chamfer_loss
            
        def jac_fn(params, *args):
            return compute_jacobian(params, *args)
        
        def hess_fn(params, *args):
            return compute_hessian(params, *args)
        
        # with torch.no_grad():
        #     if (self.init_guess_params == 0).all():
        #         self.init_guess_params[:, 0] = 1.0
        #         self.init_guess_params += 0.01*torch.randn_like(self.init_guess_params)

        #         # we will scipy minimize the error to get the best rotation

        #         # result = minimize(compute_error,
        #         #           self.init_guess_params.view(-1).detach().cpu().numpy(),
        #         #           args=(X[:, :, :3], mesh, True),
        #         #           method='L-BFGS-B',
        #         #           options={'disp': True, 'maxiter': 30},)
        #         result = differential_evolution(
        #             compute_error,
        #             bounds=[(-1, 1)] * (4 * self.batch_size),  # Quaternion bounds
        #             args=(X[:, :, :3], mesh, True),
        #             atol=1e-3,
        #             maxiter=3,
        #             popsize=16,
        #             disp=True
        #         )
        #         self.init_guess_params = result.x.reshape(self.batch_size, 4)
        
        N_max = 4
        #zms_a, scales_a, max_a = compute_zernike_moments(X[:, :, :3], N_max)

        mesh_ptcloud = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)[None, :, :]
        #zms_b, scales_b, max_b = compute_zernike_moments(mesh_ptcloud, N_max)

        #moment_dif = zms_a - zms_b
        #scales_dif = scales_a - scales_b
        #max_dif = max_a - max_b
        #moment_weights = torch.tensor([1/(i+1)**3 for i in range(moment_dif.shape[1])], dtype=torch.float32).to(self.device)

        #pdb.set_trace()
        #errors = torch.norm(moment_dif*moment_weights, dim=1).mean()
        #pdb.set_trace()
        errors = compute_error(self.init_guess_params, X[:, :, :3], mesh)
            # Compute the error
        return 5*errors, torch.zeros(1) #5*scales_dif**2 + max_dif**2
    
    def neighbour_loss(self, X, desired_neighbor_dist=1.75):
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)

        # Mask out the diagonal and only consider the (num_neighbors) closest neighbors
        neighbor_mask = cell_distances.argsort(dim=2).argsort(dim=2) < 3
        diag_mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)

        neighbor_penalty = (desired_neighbor_dist - cell_distances)**2 * neighbor_mask
        #neighbor_penalty = 1/(cell_distances**2 + 1e-3) * neighbor_mask
        neighbor_penalty[diag_mask] = 0.0

        neighbor_penalty = torch.sum(neighbor_penalty, dim=(1, 2))

        return 0.05*neighbor_penalty

    def sphere_loss(self,
                    X,
                    R=3.0,
                    nonsphere_coeff=1.0,
                    neighbor_coeff=0.05,
                    beat_heart=True,
                    num_neighbors=3,
                    desired_neighbor_dist=1.75):
        
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)  # [batch, m, m]

        desired_r = R + self.heartbeat_signal(self.t, amplitude=1.0, period=10) if beat_heart else R

        nonsphere_penalty = ((cell_positions.norm(dim=-1) - desired_r)**2).mean(dim=-1)

        # Mask out the diagonal and only consider the (num_neighbors) closest neighbors
        neighbor_mask = cell_distances.argsort(dim=2).argsort(dim=2) < num_neighbors+1
        diag_mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)

        neighbor_penalty = (desired_neighbor_dist - cell_distances)**2 * neighbor_mask
        neighbor_penalty[diag_mask] = 0.0

        #print("Desired r: ", desired_r)
        #print("Average distance from origin: ", cell_positions.norm(dim=-1).mean())
        #print("Average neighbor distance: ", cell_distances[~diag_mask & neighbor_mask].mean())

        neighbor_penalty = torch.sum(neighbor_penalty, dim=(1, 2))

        return nonsphere_coeff * nonsphere_penalty, neighbor_coeff * neighbor_penalty

    def elipse_loss(self,
                    X,
                    R=3.0,
                    nonsphere_coeff=1.0,
                    neighbor_coeff=0.05,
                    beat_heart=True):
        
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)  # [batch, m, m]

        desired_r = R #+ self.heartbeat_signal(self.t, amplitude=1.0, period=10) if beat_heart else R
        cell_positions = torch.cat((cell_positions[:, :, 0:1]*0.5, cell_positions[:, :, 1:2], cell_positions[:, :, 2:3]*2.0), dim=-1)
        nonsphere_penalty = ((cell_positions.norm(dim=-1) - desired_r)**2).mean(dim=-1)

        return nonsphere_coeff * nonsphere_penalty


    def compute_vector_scalar_correlation(self, X: torch.Tensor, morphogen_idx: int = 7) -> torch.Tensor:
        """
        Compute rotation-invariant correlation between position vectors and morphogen values for each batch.
        Args:
            X: Tensor of shape (batch, m, features) containing batched data
            morphogen_idx: Index of the morphogen value in the features dimension
        Returns:
            correlation: Tensor of shape (batch,) containing the rotation-invariant correlation for each batch
        """
        # Extract positions and morphogen values
        #pdb.set_trace()
        cell_positions = X[:, :, :3]                    # [batch, m, 3]
        morphogen_vals = X[:, :, morphogen_idx]         # [batch, m]
        # Compute w using pseudo-inverse for each batch
        # (X^T X)^(-1) X^T m
        XtX = torch.bmm(cell_positions.transpose(1, 2), cell_positions)  # [batch, 3, 3]
        Xtm = torch.bmm(cell_positions.transpose(1, 2),
                        morphogen_vals.unsqueeze(-1))   # [batch, 3, 1]
        
        # Solve XtX @ w = Xtm for each batch
        w = torch.linalg.solve(XtX, Xtm)              # [batch, 3, 1]
        # Compute predicted values
        m_hat = torch.bmm(cell_positions, w).squeeze(-1)  # [batch, m]
        # Center the variables for each batch
        m_centered = morphogen_vals - morphogen_vals.mean(dim=1, keepdim=True)
        m_hat_centered = m_hat - m_hat.mean(dim=1, keepdim=True)
        # Compute correlation for each batch
        numerator = (m_centered * m_hat_centered).sum(dim=1)
        denominator = torch.sqrt(
            (m_centered ** 2).sum(dim=1) *
            (m_hat_centered ** 2).sum(dim=1)
        )
        correlation = numerator / denominator
        # print("*******")
        # print(correlation.min())
        # print(correlation.max())
        # print(morphogen_vals.min())
        # print(morphogen_vals.max())
        # print("*******")
        return 5*correlation**2

    def heartbeat_signal(self, t, amplitude=1.0, period=29.23):
        # Returns a heartbeat-like signal, given a timestep
        # Adapted from here: https://www.desmos.com/calculator/2bqvtdd6vd
        # return amplitude * 0.3116*(
        #         (math.sin((9.4248/period)*t * 4) + (math.sin((9.4248/period)*t * 16) / 4)) * 3 *
        #         (-(math.floor(math.sin((9.4248/period)*t * 2)) + 0.1)) *
        #         (1 - math.floor(math.fmod(math.sin((9.4248/period)*t / 1.5), 2))))

        return math.sin(2*math.pi/period*t) * amplitude
    
    # These functions are for manim (animation software used for visualizing the simulation)
    def get_vertices(self):
        return [v for v in range(self.X.shape[1])]
    
    def get_edges(self, connected_threshold=2.0):
        vertex_positions = self.X[0, :, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        num_neighbors = 3
        neighbor_mask = vertex_distances.argsort(dim=1).argsort(dim=1) < num_neighbors+1

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if neighbor_mask[i, j]:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[0, i, 0:3].detach().numpy() for i in range(self.m)}

    def get_colours(self):
        # Normalize the 7th component to [0, 1] for color mapping
        color_values = self.X[0, :, 7].detach().numpy()

        # Take PCA to find dim of greatest variation for the colour values
        #pca = PCA(n_components=1)
        #color_values = pca.fit_transform(self.X[0, :, 6:].detach().numpy())

        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
        # subtract mean from color values
        #color_values = color_values - color_values.mean(dim=0, keepdims=True) + 0.5
        
        # Apply a colormap (e.g., viridis) to get RGBA values for each vertex
        cmap = plt.cm.viridis
        colors = cmap(color_values)
        
        # Return colors as a list of RGBA values
        return [colors[i] for i in range(self.m)]
    
    def write_obj(self, mesh=None):

        points = self.X[0, :, 0:3].detach().numpy()
        filename = os.path.join(self.obj_out_dir, f'point_cloud_{self.t:04d}.obj')
        with open(filename, 'w') as f:
            v_ind = 1
            for p in points:
                x, y, z = p.tolist()
                f.write(f'v {x} {y} {z}\n')
                v_ind += 1
            f.write('v 1.0 0.2 0.0\n')
            v_ind += 1

            if mesh is not None and self.t >= 470:
                # init_guess = self.init_guess_params[0]
                # if (init_guess == 0).all():
                #     self.obj_loss(self.X, mesh)
                # init_guess = self.init_guess_params[0]
                # rot_angles = init_guess[:3]
                # translation = init_guess[3:]
                # rotation_matrix = torch.tensor(R.from_euler('xyz', rot_angles).as_matrix(), dtype=torch.float).to(self.device)
                # transformed_points = (rotation_matrix @ points.T).T #+ translation

                # pdb.set_trace()
                for vertex in mesh.vertices:
                    f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
                for face in mesh.faces:
                    f.write(f'f {face[0]+v_ind} {face[1]+v_ind} {face[2]+v_ind}\n')
                