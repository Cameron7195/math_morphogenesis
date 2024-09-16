import numpy as np
import torch
import torch.nn as nn
import copy
import pdb
import math

NOISE = 0.02

tau = 1

class FourierRandomFeatures3D(nn.Module):
    def __init__(self, embedding_dim, length_scale):
        """
        Initialize the 3D Fourier Random Features embedding module.

        Args:
            embedding_dim (int): The dimension of the positional embedding.
            length_scale (float): The characteristic length scale (sigma) of interactions.
        """
        super(FourierRandomFeatures3D, self).__init__()

        assert embedding_dim % 2 == 0, "Embedding dimension must be divisible by 2."
        self.embedding_dim = embedding_dim
        self.num_length_scales = 2
        self.num_freqs_per_scale = embedding_dim // (2 * self.num_length_scales)

        # Generate a range of length scales (logarithmically spaced)
        self.length_scales = np.logspace(
            np.log10(0.1), np.log10(3.0), num=self.num_length_scales
        )

        # Generate random weights and biases for the Fourier features for each length scale
        self.W = torch.cat([
            torch.randn(self.num_freqs_per_scale, 3) * (1.0 / length_scale)
            for length_scale in self.length_scales
        ], dim=0)  # Combined tensor of shape (num_length_scales * num_freqs_per_scale, 3)

        self.b = torch.cat([
            torch.rand(self.num_freqs_per_scale) * 2 * math.pi
            for _ in range(self.num_length_scales)
        ], dim=0)  # Combined tensor of shape (num_length_scales * num_freqs_per_scale,)

    def forward(self, positions):
        """
        Apply 3D Fourier Random Features to the input positions.

        Args:
            positions (Tensor): Tensor of shape (N, 3) containing N 3D positions.

        Returns:
            Tensor: Positional embeddings of shape (N, embedding_dim).
        """
        # Project positions using random weights
        projected_positions = torch.matmul(positions, self.W.T) + self.b  # Shape: (N, num_freqs)

        # Compute Fourier features using cosines and sines
        fourier_features = torch.cat([torch.cos(projected_positions), torch.sin(projected_positions)], dim=-1)

        # Normalize the output by sqrt(2 / embedding_dim)
        return fourier_features * math.sqrt(2.0 / self.num_freqs_per_scale)

class PositionalVelocityEmbedding3D(nn.Module):
    def __init__(self, embedding_dim, length_scale):
        """
        Initialize the 3D positional and velocity embedding module.

        Args:
            embedding_dim (int): The dimension of each positional or velocity embedding.
            length_scale (float): The characteristic length scale of interactions.
            velocity_scale (float): The characteristic velocity scale of interactions.
        """
        super(PositionalVelocityEmbedding3D, self).__init__()

        assert embedding_dim % 2 == 0, "Embedding dimension must be divisible by 6"
        self.embedding_dim = embedding_dim
        self.length_scale = length_scale

        # Calculate the number of frequency components per dimension
        self.num_freqs = embedding_dim // 6

        # Define periods centered around the given length and velocity scales
        self.length_periods = self.length_scale * torch.exp(
            torch.linspace(-3, 1, self.num_freqs)
        )


    def forward(self, positions):
        """
        Apply 3D positional and velocity embedding to the input positions and velocities.

        Args:
            positions (Tensor): Tensor of shape (N, 3) containing N 3D positions.
            velocities (Tensor): Tensor of shape (N, 3) containing N 3D velocity vectors.

        Returns:
            Tensor: Combined positional embeddings of shape (N, embedding_dim).
        """
        pos_embedding = self._encode(positions, self.length_periods)
        return pos_embedding

    def _encode(self, values, periods):
        """
        Encode the input values (positions or velocities) with sinusoidal functions.

        Args:
            values (Tensor): Tensor of shape (N, 3) containing 3D vectors.
            periods (Tensor): Tensor containing period components.

        Returns:
            Tensor: Encoded values of shape (N, embedding_dim).
        """
        # Extract x, y, z components
        x, y, z = values[:, :, 0:1], values[:, :, 1:2], values[:, :, 2:3]

        # Compute sine and cosine encodings for each component using periods
        encodings = []
        for period in periods:
            # Convert period to frequency (inverse of period)
            freq = 2*math.pi / period
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.sin(freq * y))
            encodings.append(torch.sin(freq * z))
            encodings.append(torch.cos(freq * x))
            encodings.append(torch.cos(freq * y))
            encodings.append(torch.cos(freq * z))
        # Concatenate encodings
        return torch.cat(encodings, dim=-1)    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q (Tensor): Query tensor of shape (batch_size, n_query, d_model).
            K (Tensor): Key tensor of shape (batch_size, n_key, d_model).
            V (Tensor): Value tensor of shape (batch_size, n_key, d_model).
            mask (Tensor, optional): Attention mask of shape (batch_size, n_heads, n_query, n_key). Default is None.

        Returns:
            Tensor: Output tensor of shape (batch_size, n_query, d_model).
            Tensor: Attention scores of shape (batch_size, n_heads, n_query, n_key).
        """
        batch_size = Q.size(0)

        # Linearly project queries, keys, and values
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # Split the d_model dimension into n_heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        # Transpose to perform attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5

        # Apply the attention mask (if provided)
        if mask is not None:
            mask_expanded = mask[:, None, :, :].repeat(1, self.n_heads, 1, 1)
            masked_scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Compute attention weights
        weights = torch.softmax(masked_scores, dim=-1)

        # Apply attention weights to values
        attention = torch.matmul(weights, V)

        # Concatenate attention heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Linearly project output
        return self.W_O(attention), scores
    
class MorphoMLP(nn.Module):
    def __init__(self, d_model):
        super(MorphoMLP, self).__init__()
        self.d_model = d_model
        self.intermediate_size = 4*d_model

        self.uproj = nn.Linear(d_model, self.intermediate_size)
        self.dproj = nn.Linear(self.intermediate_size, d_model)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.uproj(x)
        x = self.act(x)
        x = self.dproj(x)
        return x

# force_net should output, for pairwise x_i, x_j: k, L. That's it! Two numbers!
class force_net(torch.nn.Module):
    def __init__(self, n, d_model, n_heads, n_layers, device='cpu'):
        super(force_net, self).__init__()
        self.n = n
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.length_scale = 1.8
        self.velocity_scale = 1.0
        self.device = device

        # enforce d_model multiple of 6
        #assert self.d_model % 6 == 0
        
        #self.pos_embed = PositionalVelocityEmbedding3D(self.d_model, self.length_scale)
        self.pos_embed = FourierRandomFeatures3D(self.d_model, self.length_scale)
        self.char_embed = torch.nn.Linear(n - 6, self.d_model)

        # Initialize n_layers of multi-head attention
        self.attentions = nn.ModuleList([MultiHeadAttention(self.d_model, self.n_heads) for _ in range(self.n_layers)])

        # Initialize layer norms
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(2*self.n_layers + 1)])
        self.score_norm = nn.LayerNorm(self.n_heads)

        # Initialize morpho MLP
        self.morpho_mlps = nn.ModuleList([MorphoMLP(self.d_model) for _ in range(self.n_layers)])

        self.final_proj = nn.Linear(self.d_model, n-6)

        self.scores_uproj = nn.Linear(self.n_heads+1, self.n_heads*16)
        self.scores_h = nn.Linear(self.n_heads*16, self.n_heads*16)
        self.scores_dproj = nn.Linear(self.n_heads*16, 1)
        self.scores_final_proj = nn.Linear(self.n_heads, 1)
        self.act = nn.GELU()

        init_params = torch.randn(1, n-6)
        init_params = init_params / (torch.norm(init_params, dim=1, keepdim=True) + 1e-8)
        self.init_state_params = torch.nn.Parameter(init_params, requires_grad=True)


    def forward(self, X, mask=None):
        # 
        batch_size = X.shape[0]
        m = X.shape[1]
        positions = X[:, :, 0:3]
        velocities = X[:, :, 3:6]

        real_displacements = positions[:, None, :, :] - positions[:, :, None, :] # (batch x m x m x 3)
        real_distances = torch.norm(real_displacements, dim=3)

        pos_embeddings = self.pos_embed(positions)

        # Normalize x inputs
        x = self.char_embed(X[:, :, 6:])

        x = x + pos_embeddings
        
        # Implement MHA
        for i, attention in enumerate(self.attentions):
            x_norm = self.layer_norms[i](x)
            x_att, scores = attention(x_norm, x_norm, x_norm, mask)
            x = x + x_att

            x_norm = self.layer_norms[i + self.n_layers](x)
            x_mlp = self.morpho_mlps[i](x_norm)
            x = x + x_mlp

        # one last norm
        x_norm = self.layer_norms[-1](x)
        x = self.final_proj(x_norm)

        scores = scores.permute(0, 2, 3, 1)
        #scores = torch.cat([scores, real_distances.unsqueeze(-1)], dim=-1)
        #scores = self.score_norm(scores)
        #scores = self.scores_uproj(scores)
        #scores = self.act(scores)
        #scores = self.scores_h(scores)
        scores = self.score_norm(scores)
        #scores = self.act(scores)
        #scores = self.scores_dproj(scores).squeeze(-1)
        scores = self.scores_final_proj(scores).squeeze(-1)
        scores = (scores.permute(0, 2, 1) + scores)/2

        return x, scores

class Organism():
    def __init__(self, n, fn, batch_size=1):
        if n < 6:
            raise ValueError("Cell State must have at least 6 dimensions (position and velocity).")
        

        self.fn = fn
        self.device = fn.device
        self.batch_size = batch_size
        self.X = torch.cat((torch.zeros(batch_size, 1, 6).to(self.device), self.fn.init_state_params.repeat(batch_size, 1, 1)), dim=-1)
        
        # Add noise to initial state, excluding position and velocity
        self.X[:, :, 6:] += NOISE*torch.randn_like(self.X[:, :, 6:])

        self.n = n
        self.m = 1
        self.t = 0.0
        self.dt = 0.2
        self.b = 4.0

        # Conc[0] will be mitogenic factor
        self.dt = 0.1

    def evolve(self):
        # Euler
        new_X = self.X + self.dt*self.f(self.X)
        x_h_norm = new_X[:, :, 6:] / (torch.norm(new_X[:, :, 6:], dim=2, keepdim=True) + 1e-8)
        self.X = torch.cat((new_X[:, :, 0:6], x_h_norm), dim=2)

        self.divide(self.X)
        self.t += 1
        #print(self.t)
        #print(self.m)
        return
    
    def f(self, X):
        X_prime = torch.zeros((self.batch_size, self.m, self.n)).to(self.device)

        X_prime[:, :, 0:3] = X[:, :, 3:6]

        mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)
        vertex_positions = X[:, :, 0:3]
        vertex_displacements = vertex_positions[:, None, :, :] - vertex_positions[:, :, None, :] # (batch x m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=3)
        vertex_unit_vectors = -vertex_displacements/(vertex_distances[:, :, :, None]+1e-6)

        # assert not nan
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(vertex_unit_vectors))
        vertex_unit_vectors[mask] = 0.0

        attention_mask = torch.ones(self.batch_size, self.m, self.m).to(self.device)
        attention_mask[vertex_distances > 2.0] = 0.0

        Xp, k = self.fn(X, mask) # (batch x m x m), (batch x m x m x n)

        # if self.t % 100 == 0:
        #     print("t = ", self.t)
        #     print('X mean: ', torch.mean(X[:, :, 6:]))
        #     print('X max: ', torch.max(X[:, :, 6:]))
        #     print('X min: ', torch.min(X[:, :, 6:]))
        #     print("k mean: ", torch.mean(k))
        #     print("k max: ", torch.max(k))
        #     print("k min: ", torch.min(k))

        # Add noise to k and Xp
        k += NOISE*torch.randn_like(k)
        Xp += NOISE*torch.randn_like(Xp)

        spring_force = k[:, :, :, None] * vertex_unit_vectors

        # Find orthogonal vectors to the unit vectors, rotated by angle theta
    
        spring_force[mask] = 0.0
        spring_force[vertex_distances > 2.0] = 0.0

        spring_force = torch.sum(spring_force, dim=2)

        velocities = X[:, :, 3:6]

        friction_force = -self.b*velocities

        X_prime[:, :, 3:6] = spring_force + friction_force

        X_prime[:, :, 6:] = Xp

        return X_prime
    
    def divide(self, X):
        
        if self.m < 32 and self.t % 80 == 79:
            #i = torch.randint(0, self.m, (1,)).item()
            new_xi = X + torch.rand(self.batch_size, self.m, self.n).to(self.device)/100
            self.X = torch.cat((self.X, new_xi), dim=1)
            self.m *= 2

        return
    
    def sphere_loss(self, X):
        vertex_positions = X[:, :, 0:3]
        # if self.m > 2:
        #     dist_variation = torch.var(torch.norm(vertex_positions, dim=2), dim=1)
        # else:
        #     dist_variation = torch.zeros(self.batch_size).to(self.device)

        mask = torch.eye(self.m).bool().repeat(self.batch_size, 1, 1).to(self.device)
        vertex_positions = X[:, :, 0:3]
        vertex_displacements = vertex_positions[:, None, :, :] - vertex_positions[:, :, None, :] # (batch x m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=3)

        desired_r = 5.0
        dist_variation = ((vertex_positions.norm(dim=-1) - desired_r)**2).mean(dim=-1)

        distance_mask = vertex_distances.argsort(dim=2).argsort(dim=2) < 4

        closeness_penalty = (1.2 - vertex_distances)**2 * distance_mask

        closeness_penalty[mask] = 0.0
        closeness_penalty = torch.sum(closeness_penalty, dim=(1, 2))

        regularization = torch.sum(X[:, :, 6:]**2, dim=(1, 2))
        # if self.t % 50 and self.m > 8:
        #     pdb.set_trace()

        # Get indices of all vertices that are not within 2.0 of ANY other vertex
        if self.m == 1:
            loner_penalty = torch.zeros(self.batch_size).to(self.device)
        else:
            threshold = 2.0
            too_far_mask = (vertex_distances < threshold) & ~mask
            loner_mask = torch.all(~too_far_mask, dim=2)  # Shape: (batch_size, m), True where vertex is a loner
            nearest_distances = torch.min(vertex_distances + 1e6*mask, dim=2).values
            loner_penalty = torch.sum(nearest_distances**2 *loner_mask, dim=1)


        loner_penalty = loner_penalty*0


        return dist_variation, regularization, closeness_penalty, loner_penalty


    def exploding(self):
        if torch.any(torch.abs(self.X) > 50):
            return True
        return False
    
    def get_vertices(self):
        return [v for v in range(self.X.shape[1])]
    
    def get_edges(self):
        vertex_positions = self.X[0, :, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2).detach().numpy()

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if vertex_distances[i, j] < 2.0:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[0, i, 0:3].detach().numpy() for i in range(self.m)}