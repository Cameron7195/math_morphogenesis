# Imports
import numpy as np
import torch
import torch.nn as nn
import copy
import pdb
import math
from equiformer_pytorch import Equiformer
from se3_transformer_pytorch import SE3Transformer
from scipy.spatial.transform import Rotation as R
import trimesh
import os
from torch.autograd import grad
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt
from skimage import color

from matplotlib.colors import Normalize, hsv_to_rgb

from PIL import Image



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
                                           num_neighbors=5,
                                           num_linear_attn_heads = 0)
        
        # self.se3_tr_model = SE3Transformer(dim=d_model,
        #                                    dim_in=self.input_dim,
        #                                    dim_head=d_model//n_heads,
        #                                    heads=n_heads,
        #                                    num_neighbors=6,
        #                                    num_degrees=2,
        #                                    depth=n_layers)

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
                 noise=0.0001,
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
        self.prev_X = self.X
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
            os.makedirs(self.obj_out_dir + "_6", exist_ok=True)
            os.makedirs(self.obj_out_dir + "_7", exist_ok=True)
            os.makedirs(self.obj_out_dir + "_8", exist_ok=True)
            self.write_obj(out_dir=self.obj_out_dir + "_6", morphogen_idx=6)
            self.write_obj(out_dir=self.obj_out_dir + "_7", morphogen_idx=7)
            self.write_obj(out_dir=self.obj_out_dir + "_8", morphogen_idx=8)

        self.squared_vels = 0
        self.squared_state_vels = 0


    def evolve(self):
        # Euler method
        self.prev_X = self.X
        fx = self.f(self.X)
        self.X = self.X + self.dt*fx

        self.squared_vels += (fx[:, :, 0:3].norm(dim=-1)**2).sum(dim=1)
        self.squared_state_vels += (fx[:, :, 6:].norm(dim=-1)**2).sum(dim=1)

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
            self.write_obj(out_dir=self.obj_out_dir + "_6", morphogen_idx=6)
            self.write_obj(out_dir=self.obj_out_dir + "_7", morphogen_idx=7)
            self.write_obj(out_dir=self.obj_out_dir + "_8", morphogen_idx=8)
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
        
        if self.m < 128 and self.t % 60 == 1:
            new_xi = X
            # if self.m <= 2:
            #     new_xi = new_xi + 0.1*self.noise*torch.ones_like(X).to(self.device) # [batch, 1, n]
            # else:
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
            idx = torch.randperm(pc2.shape[1])[:self.m]
            pc2 = pc2[:, idx, :]
            pc2 = pc2.repeat(pc1.shape[0], 1, 1)

            # First, we'll centre the point cloud
            pc1 = pc1 - pc1.mean(dim=1, keepdim=True)
            pc2 = pc2 - pc2.mean(dim=1, keepdim=True)

            
            # Compute sum of all squared distances

            rel_positions = pc1[:, :, None, :] - pc2[:, None, :, :]
            rel_l2_squared = torch.norm(rel_positions, dim=3)**2
            total_loss = rel_l2_squared.sum(dim=(1, 2))
            return total_loss
            
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
        return 0.0004*errors
    
    def neighbour_loss(self, X, desired_neighbor_dist=0.9):
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)

        # Mask out the diagonal and only consider the (num_neighbors) closest neighbors
        num_neighbors = 4
        neighbor_mask = cell_distances.argsort(dim=2).argsort(dim=2) < num_neighbors + 1
        diag_mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)
        desired_neighbor_dist = desired_neighbor_dist * torch.exp(X[:, :, 7].unsqueeze(-1))

        max_morphogen = X[:, :, 7].max(dim=1, keepdim=True)[0]
        min_morphogen = X[:, :, 7].min(dim=1, keepdim=True)[0]
        morphogen_range = max_morphogen - min_morphogen
        middle_morphogen = (max_morphogen + min_morphogen) / 2
        desired_neighbor_dist = 1.5 #* torch.ones_like(cell_positions.norm(dim=-1))
        # Top 10% of cells in morphogen conc
        #desired_neighbor_dist[X[:, :, 7] > middle_morphogen + 0.4*morphogen_range] = 0.7
        #desired_neighbor_dist = desired_neighbor_dist * ((X[:, :, 7] - min_morphogen)/morphogen_range + 0.05)
        #desired_neighbor_dist = (desired_neighbor_dist[:, :, None] + desired_neighbor_dist[:, None, :])/2

        neighbor_penalty = (desired_neighbor_dist - cell_distances)**2 * neighbor_mask
        #neighbor_penalty = 1/(cell_distances**2 + 1e-3) * neighbor_mask
        neighbor_penalty[diag_mask] = 0.0

        neighbor_penalty = torch.sum(neighbor_penalty, dim=(1, 2))

        return 0.02*neighbor_penalty

    def sphere_loss(self,
                    X,
                    R=3.0,
                    nonsphere_coeff=3.0,
                    neighbor_coeff=0.05,
                    beat_heart=False,
                    num_neighbors=3,
                    desired_neighbor_dist=1.75):
        
        cell_positions = X[:, :, 0:3]                           # [batch, m, 3]
        cell_displacements =   cell_positions[:, None, :, :] \
                             - cell_positions[:, :, None, :]    # [batch, m, m, 3]
        cell_distances = torch.norm(cell_displacements, dim=3)  # [batch, m, m]

        desired_r = R + self.heartbeat_signal(self.t, amplitude=1.0, period=10) if beat_heart else R

        max_morphogen = X[:, :, 7].max(dim=1, keepdim=True)[0]
        min_morphogen = X[:, :, 7].min(dim=1, keepdim=True)[0]
        morphogen_range = max_morphogen - min_morphogen
        middle_morphogen = (max_morphogen + min_morphogen) / 2
        desired_r = 4.0 * torch.ones_like(cell_positions.norm(dim=-1))
        # Top 10% of cells in morphogen conc
        desired_r = desired_r * ((X[:, :, 7] - min_morphogen)/morphogen_range + 0.1)
        
       # print(desired_r)

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

        return nonsphere_coeff * nonsphere_penalty

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


    def compute_vector_scalar_correlation(self, X: torch.Tensor, morphogen_idx: int = 7, radial=False, override_w = None, project1=None, project2=None) -> torch.Tensor:
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
        if radial == True:
            cell_positions = torch.norm(cell_positions, dim=-1, keepdim=True)  # [batch, m, 1]
        if project1 != None:
                        # Ensure w1_unit is of shape [batch, 3]
            w1_unit = project1 / torch.norm(project1, dim=-1, keepdim=True)  # [batch, 3]

            # Compute dot product between cell_positions and w1_unit
            dot_products = torch.sum(cell_positions * w1_unit[:, None, :], dim=2, keepdim=True)  # [batch, m, 1]

            # Project onto plane
            cell_positions = cell_positions - dot_products * w1_unit[:, None, :]  # [batch, m, 3]
            #residual = torch.sum(cell_positions * w1_unit[:, None, :, 0], dim=2)  # [batch, m]
            #print(residual)
            if project2 != None:
                    # Ensure w2_unit is of shape [batch, 3]
                w2_unit = project2 / torch.norm(project2, dim=-1, keepdim=True)  # [batch, 3]

                # Compute dot product between cell_positions and w2_unit_in_plane
                dot_products = torch.sum(cell_positions * w2_unit[:, None, :], dim=2, keepdim=True)  # [batch, m, 1]

                # Project onto line orthogonal to w2_unit_in_plane
                cell_positions = cell_positions - dot_products * w2_unit[:, None, :]  # [batch, m, 3]
        if override_w is not None:
            # project onto the line defined by w
            w_unit = override_w / torch.norm(override_w, dim=-1, keepdim=True)
            dot_products = torch.sum(cell_positions * w_unit[:, None, :], dim=2, keepdim=True)
            cell_positions = dot_products

        morphogen_vals = X[:, :, morphogen_idx]         # [batch, m]
        m_centered = morphogen_vals - morphogen_vals.mean(dim=1, keepdim=True)
        # Compute w using pseudo-inverse for each batch
        # Prepare X and y for least squares
        y = m_centered.unsqueeze(-1)  # [batch, m, 1]

        w = torch.linalg.lstsq(cell_positions, y).solution


        # Compute predicted values
        m_hat = torch.bmm(cell_positions, w).squeeze(-1) # [batch, m]

        # Center the variables for each batch
        
        m_hat_centered = m_hat - m_hat.mean(dim=1, keepdim=True)
        # Compute correlation for each batch
        numerator = (m_centered * m_hat_centered).sum(dim=1)
        denominator = torch.sqrt(
            (m_centered ** 2).sum(dim=1) *
            (m_hat_centered ** 2).sum(dim=1)
        )
        correlation = numerator / denominator
        return 20*correlation.abs(), w[:, :, 0]
    
    def compute_total_morphogen_correlation(self, X: torch.Tensor, morphogen_indices: list = [6, 7, 8]) -> torch.Tensor:
        """
        Compute rotation-invariant correlation between position vectors and morphogen values for each batch,
        using orthogonal basis vectors.

        Args:
            X: Tensor of shape (batch, m, features) containing batched data
            morphogen_indices: List of indices of the morphogen values in the features dimension

        Returns:
            total_correlation: Tensor of shape (batch,) containing the total correlation for each batch
        """
        # Extract positions and morphogen values
        cell_positions = X[:, :, :3]  # [batch, m, 3]
        morphogen_vals = X[:, :, morphogen_indices]  # [batch, m, 3]

        # Compute W using least squares for each batch
        # W = (X^T X)^(-1) X^T M
        XtX = torch.bmm(cell_positions.transpose(1, 2), cell_positions)  # [batch, 3, 3]
        XtM = torch.bmm(cell_positions.transpose(1, 2), morphogen_vals)  # [batch, 3, 3]

        # Solve for W: XtX W = XtM
        W = torch.linalg.solve(XtX, XtM)  # [batch, 3, 3]

        # Perform Gram-Schmidt orthogonalization on the columns of W for each batch
        def gram_schmidt(W_batch):
            # W_batch: [3, 3]
            Q_list = []
            for i in range(3):
                v = W_batch[:, i]  # [3]
                for j in range(i):
                    qj = Q_list[j]  # [3]
                    # Compute the projection of v onto qj
                    proj = (torch.dot(v, qj) / torch.dot(qj, qj)) * qj
                    # Subtract the projection from v to make it orthogonal to qj
                    v = v - proj
                Q_list.append(v)
            # Stack the orthogonal vectors to form Q_batch
            Q_batch = torch.stack(Q_list, dim=1)  # [3, 3]
            return Q_batch

        # Apply Gram-Schmidt orthogonalization batch-wise
        W_orth = torch.stack([gram_schmidt(W[b]) for b in range(W.shape[0])], dim=0)  # [batch, 3, 3]

        # Compute predicted morphogen values
        M_hat = torch.bmm(cell_positions, W_orth)  # [batch, m, 3]

        # Center the variables for each batch
        M_centered = morphogen_vals - morphogen_vals.mean(dim=1, keepdim=True)
        M_hat_centered = M_hat - M_hat.mean(dim=1, keepdim=True)

        # Flatten M_hat_centered and M_centered along m and features dimensions
        M_hat_flat = M_hat_centered.reshape(M_hat_centered.shape[0], -1)  # [batch, m*3]
        M_centered_flat = M_centered.reshape(M_centered.shape[0], -1)      # [batch, m*3]

        # Compute correlation for each batch
        numerator = (M_hat_flat * M_centered_flat).sum(dim=1)
        denominator = torch.sqrt(
            (M_hat_flat ** 2).sum(dim=1) * (M_centered_flat ** 2).sum(dim=1)
        )
        correlation = numerator / denominator  # [batch,]

        # Return the absolute value of the correlation scaled as per original code
        return 20 * correlation.abs()
    
    def compute_morphogen_correlation(self, X: torch.Tensor, morphogen_idx1: int, morphogen_idx2: int) -> torch.Tensor:
        """
        Compute correlation between two morphogen concentrations for each batch.
        Args:
            X: Tensor of shape (batch, m, features) containing batched data
            morphogen_idx1: Index of the first morphogen value in the features dimension
            morphogen_idx2: Index of the second morphogen value in the features dimension
        Returns:
            correlation: Tensor of shape (batch,) containing the correlation between the two morphogens for each batch
        """
        # Extract morphogen values for the specified indices
        morphogen_vals1 = X[:, :, morphogen_idx1]  # [batch, m]
        morphogen_vals2 = X[:, :, morphogen_idx2]  # [batch, m]

        # Center the variables for each batch
        m1_centered = morphogen_vals1 - morphogen_vals1.mean(dim=1, keepdim=True)
        m2_centered = morphogen_vals2 - morphogen_vals2.mean(dim=1, keepdim=True)

        # Compute correlation for each batch
        numerator = (m1_centered * m2_centered).sum(dim=1)
        denominator = torch.sqrt(
            (m1_centered ** 2).sum(dim=1) *
            (m2_centered ** 2).sum(dim=1)
        )
        correlation = numerator / denominator

        return 20*correlation.abs()  # Absolute value to focus on correlation magnitude

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
    
    def get_edges(self, connected_threshold=1.4):
        vertex_positions = self.X[0, :, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        num_neighbors = 4
        neighbor_mask = vertex_distances.argsort(dim=1).argsort(dim=1) < num_neighbors+1
        #neighbor_mask = vertex_distances < connected_threshold

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if neighbor_mask[i, j]:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[0, i, 0:3].detach().numpy() for i in range(self.m)}

    def get_colours(self, morphogen_idx=6, use_hsv=False):
        # Normalize the selected morphogen component to [0, 1] for color mapping
        color_values = self.X[0, :, morphogen_idx].detach().numpy()

        # Apply normalization to the color values
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        normalized_values = norm(color_values)

        if (self.t // 10) % 2 == 0:
            morphogen_idx = 6
        elif (self.t // 10) % 2 == 1:
            morphogen_idx = 7
        elif (self.t // 10) % 4 == 2:
            morphogen_idx = 8
        else:
            morphogen_idx = 9

        # Map morphogens to Lab components if use_lab is True
        if use_hsv:
            light_values = self.X[0, :, 6].detach().numpy()
            a_values = self.X[0, :, 7].detach().numpy()
            b_values = self.X[0, :, 8].detach().numpy()

            # Normalize each morphogen component to [0, 1]
            norm_light = Normalize(vmin=light_values.min(), vmax=light_values.max())
            norm_a = Normalize(vmin=a_values.min(), vmax=a_values.max())
            norm_b = Normalize(vmin=b_values.min(), vmax=b_values.max())
            # Map each morphogen to Lab space ranges
            lightness = norm_light(light_values) * 100  # L is typically 0–100 in Lab
            a_star = norm_a(a_values) * 200 - 100       # a* is typically -100 to 100
            b_star = norm_b(b_values) * 200 - 100       # b* is typically -100 to 100

            # Combine L, a*, b* values and convert to RGB
            lab_colors = np.stack([lightness, a_star, b_star], axis=-1)
            colors = color.lab2rgb(lab_colors)  # Convert Lab to RGB for visualization

            # add alpha channel
            colors = [list(c) + [1.0] for c in colors]
        
        else:
            # Use predefined colormaps based on morphogen index
            if morphogen_idx == 6:
                cmap = plt.cm.viridis
            elif morphogen_idx == 7:
                cmap = plt.cm.plasma
            elif morphogen_idx == 8:
                cmap = plt.cm.cividis
            else:
                cmap = plt.cm.inferno

            colors = cmap(normalized_values)

        # Return colors as a list of RGBA values
        return [colors[i] for i in range(self.m)]
    
    def create_texture(self, colors, filename, target_size=516*2*4):
        """
        Creates a 1D texture map from vertex colors with a fixed size of 129.
        colors: Array of shape (N, 3) or (N, 4) with RGB or RGBA values in [0, 255].
        filename: Output texture image filename.
        target_size: Fixed texture size (default 129).
        """
        N = colors.shape[0]
        colors_uint8 = np.flip((colors[:, :3] * 255).astype(np.uint8), axis=0)
        colors_uint8_out = np.zeros((target_size, 3), dtype=np.uint8)

        # Resize color array to match target size
        # Upsample to match target size. Simply repeat 
        # num_repeats = (target_size) // (N)
        # for i in range(N):
        #     colors_uint8_out[i*num_repeats:(i+1)*num_repeats] = colors_uint8[i][None, :].repeat(num_repeats, axis=0)

        # instaead we're going to upscale colors_uint8 with a kind of 'reverse pooling'. Basically, we want to divide the output
        # image into N segments, and assign the color of each segment to the corresponding segment in the output image.
        # We will have to pay careful attention to the rounding of the indices to avoid off-by-one errors.

        # Create a mapping from the target size to the original size
        target_to_original = np.linspace(0, N, target_size, endpoint=False)
        target_to_original = np.floor(target_to_original).astype(np.int32)

        # Assign the colors to the output image
        for i in range(target_size):
            colors_uint8_out[i] = colors_uint8[target_to_original[i]]

        # Create an image of size (target_size x 1)
        texture_image = colors_uint8_out.reshape((target_size, 1, 3))
        img = Image.fromarray(texture_image, 'RGB')
        img.save(filename)


    def assign_uvs(self, N, target_size=129):
        """
        Assign UV coordinates to vertices.
        N: Number of vertices.
        Returns an array of shape (N, 2).
        """
        u_coords = np.linspace(0, 1.0, N, endpoint=True)
        v_coords = np.linspace(0, 1.0, N, endpoint=True)
        uvs = np.column_stack((u_coords, v_coords))
        return uvs

    def write_obj(self, out_dir=None, morphogen_idx=6):
        
        points = self.X[0, :, 0:3].detach().numpy()
        edges = self.get_edges()
        if morphogen_idx == 6:
            obj_filename = os.path.join(out_dir, f'pc_six_{self.t:04d}.obj')
            texture_filename = os.path.join(out_dir, f'texture_six_{self.t:04d}.png')
        elif morphogen_idx == 7:
            obj_filename = os.path.join(out_dir, f'pc_seven_{self.t:04d}.obj')
            texture_filename = os.path.join(out_dir, f'texture_seven_{self.t:04d}.png')
        elif morphogen_idx == 8:
            obj_filename = os.path.join(out_dir, f'pc_eight_{self.t:04d}.obj')
            texture_filename = os.path.join(out_dir, f'texture_eight_{self.t:04d}.png')

        N = points.shape[0]
        uvs = self.assign_uvs(N+1)
        
        colors = np.array(self.get_colours(morphogen_idx=morphogen_idx))
        colors = np.concatenate((colors, np.ones((1, 4))), axis=0)
        # Create the texture image
        colors_uint8 = (colors[:, :3] * 255).astype(np.uint8)
        self.create_texture(colors_uint8, texture_filename)

        # Write the .obj file
        with open(obj_filename, 'w') as obj_file:
            #obj_file.write(f'mtllib material_{self.t:04d}.mtl\n')
            #obj_file.write('usemtl mat0\n')
            # Write vertices
            i = 1
            for p in points:
                x, y, z = p.tolist()
                obj_file.write(f'v {x} {y} {z}\n')
                obj_file.write(f'f {i}/{i} {i}/{i} {i}/{i}\n')
                i += 1

            obj_file.write(f'v 1.0 0.2 0.0\n')

            #Write UVs
            for uv in uvs:
                u, v = uv.tolist()
                obj_file.write(f'vt {u} {v}\n')

            edges = self.get_edges()

            for edge in edges:
                i, j = edge
                obj_file.write(f'l {i+1} {j+1}\n')
                obj_file.write(f'f {i+1}/{i+1} {j+1}/{j+1} {i+1}/{i+1}\n')

            obj_file.write(f'l 1 1\n')