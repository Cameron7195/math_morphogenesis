import numpy as np
import torch
import copy
import pdb

device = torch.device("cpu")
NOISE = 0.0

tau = 1


# force_net should output, for pairwise x_i, x_j: k, L. That's it! Two numbers!
class force_net(torch.nn.Module):
    def __init__(self, n):
        super(force_net, self).__init__()
        self.n = n
        self.n_heads = 4
        self.fc1 = torch.nn.Linear(2*n+2, 8*n)
        self.fc2 = torch.nn.Linear(8*n, 8*n)
        self.fc3 = torch.nn.Linear(8*n, n+1)
        self.relu = torch.nn.GELU()
        #self.layer_norm = torch.nn.LayerNorm(4*n)
        self.init_state_params = torch.nn.Parameter(torch.randn(1, n).to(device), requires_grad=True)

    def forward(self, x, y, d):

        o = torch.ones_like(d)
        x_aug = torch.cat((x, y, d, o), dim=-1)

        z = self.fc1(x_aug)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        z = self.fc3(z)

        k = z[:, :, 0]
        Xp = z[:, :, 1:]

        return k, Xp

class Organism():
    def __init__(self, n, fn):
        if n < 6:
            raise ValueError("Cell State must have at least 6 dimensions (position and velocity).")
        

        self.fn = fn
        self.X = torch.cat((torch.zeros(1, 6).to(device), self.fn.init_state_params), dim=-1)
        
        self.n = n
        self.m = 1
        self.t = 0.0
        self.dt = 0.1
        self.b = 4.0

        # Conc[0] will be mitogenic factor
        self.dt = 0.1

    def evolve(self):
        # Euler
        self.X = self.X + self.dt*self.f(self.X)
        #self.X += NOISE * torch.randn_like(self.X)
        #self.X[:, 0:3] -= torch.mean(self.X[:, 0:3], axis=0)

        # RK
        # k1 = self.f(self.X)
        # k2 = self.f(self.X + 0.5*self.dt*k1)
        # k3 = self.f(self.X + 0.5*self.dt*k2)
        # k4 = self.f(self.X + self.dt*k3)
        # self.X = self.X + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

        self.divide(self.X)
        self.t += 1
        #print(self.t)
        #print(self.m)
        return
    
    def f(self, X):
        X_prime = torch.zeros((self.m, self.n)).to(device)

        X_prime[:, 0:3] = X[:, 3:6]

        mask = torch.eye(self.m).bool()
        vertex_positions = X[:, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2)
        vertex_unit_vectors = -vertex_displacements/(vertex_distances[:, :, None]+1e-6)

        vertex_unit_vectors[mask] = 0.0

        x = X[None, :, 6:].repeat(self.m, 1, 1)
        y = X[:, None, 6:].repeat(1, self.m, 1)
        d = vertex_distances[:, :, None]

        k, Xp = self.fn(x, y, d)
        k = (k.T + k)/2

        spring_force = k[:, :, None] * vertex_unit_vectors

        # Find orthogonal vectors to the unit vectors, rotated by angle theta
    
        spring_force[mask] = 0.0
        spring_force[vertex_distances > 2.0] = 0.0

        spring_force = torch.sum(spring_force, dim=1)

        velocities = X[:, 3:6]

        friction_force = -self.b*velocities

        X_prime[:, 3:6] = spring_force + friction_force

        Xp = torch.mean(Xp, dim=1)

        X_prime[:, 6:] = Xp

        return X_prime
    
    def divide(self, X):
        
        if self.m < 32 and self.t % 80 == 79:
            #i = torch.randint(0, self.m, (1,)).item()
            new_xi = X + torch.rand(self.m, self.n).to(device)/100
            self.X = torch.vstack((self.X, new_xi))
            self.m *= 2

        return
    
    def sphere_loss(self, X):
        vertex_positions = X[:, 0:3]
        if self.m > 2:
            dist_variation = torch.var(torch.norm(vertex_positions, dim=1))
        else:
            dist_variation = 0.0

        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        closeness_penalty = (0.8 - vertex_distances) * (vertex_distances < 0.8)
        mask = torch.eye(self.m).bool()
        closeness_penalty[mask] = 0.0
        closeness_penalty = torch.sum(closeness_penalty)

        regularization = torch.mean(X**2)
        # if self.t % 50 and self.m > 8:
        #     pdb.set_trace()

        return dist_variation + 0.1*regularization + 0.01*closeness_penalty
    
    def cylinder_loss(self, X):
        vertex_positions = X[:, 0:3]
        if self.m > 2:
            dist_variation = torch.var(torch.norm(vertex_positions[:, 0:2], dim=1))
        else:
            dist_variation = 0.0

        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :]
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        closeness_penalty = (0.8 - vertex_distances) * (vertex_distances < 0.8)

        mask = torch.eye(self.m).bool()

        closeness_penalty[mask] = 0.0

        closeness_penalty = torch.sum(closeness_penalty)

        regularization = torch.mean(X**2)

        return dist_variation + 0.05*regularization + 0.1*closeness_penalty
    
    def cube_loss(self, X):
        vertex_positions = X[:, 0:3]
        dist_variation = (torch.max(vertex_positions, dim=0).values - torch.max(vertex_positions, dim=0).values)**2
        dist_variation = torch.sum(dist_variation)

        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        closeness_penalty = (0.8 - vertex_distances) * (vertex_distances < 0.8)
        mask = torch.eye(self.m).bool()
        closeness_penalty[mask] = 0.0
        closeness_penalty = torch.sum(closeness_penalty)

        regularization = torch.mean(X**2)

        return dist_variation + 0.1*regularization + 0.01*closeness_penalty

    def skewness_loss(self, X):
        vertex_positions = X[:, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :]
        vertex_distances = torch.norm(vertex_displacements, dim=2)

        normalized_positions = vertex_positions - torch.mean(vertex_positions, dim=0)

        if self.m >= 4:
            # Calculate the covariance matrix of position
            covariance = torch.einsum('mi, mj -> ij', normalized_positions, normalized_positions)
            covariance /= self.m

            # Calculate the eigenvalues of the covariance matrix
            eigenvalues = torch.linalg.eigvalsh(covariance)
            
            # Sort eig in descending order
            eigenvalues = torch.sort(eigenvalues, descending=True).values

            skew_loss = (2* eigenvalues[0] / (eigenvalues[1] + eigenvalues[2] + 1e-6) - 8)**2

            other_loss = (eigenvalues[1] - eigenvalues[2])**2
        else:
            skew_loss = 0.0
            other_loss = 0.0

        regularization = torch.mean(X**2)

        closeness_penalty = (0.8 - vertex_distances) * (vertex_distances < 0.8)
        mask = torch.eye(self.m).bool()
        closeness_penalty[mask] = 0.0
        closeness_penalty = torch.sum(closeness_penalty)

        return 0.1*skew_loss + 0.1*other_loss + 0.1*regularization + 0.1*closeness_penalty


    def exploding(self):
        if torch.any(torch.abs(self.X) > 20):
            return True
        return False
    
    def get_vertices(self):
        return [v for v in range(self.X.shape[0])]
    
    def get_edges(self):
        vertex_positions = self.X[:, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = torch.norm(vertex_displacements, dim=2).detach().numpy()

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if vertex_distances[i, j] < 1.4:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[i, 0:3].detach().numpy() for i in range(self.m)}