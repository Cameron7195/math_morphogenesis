import numpy as np
import copy
import pdb

# To do: abstract away from even a graph representation.
# We just have a number of nodes with interaction dynamics.
# Some will have 'spring' dynamics, controlled by whether they are connected.
# What I really have to think through is the correct parameter space for this...
# Becasue some subspace of the parameters should correspond to connected nodes
# Technically I guess surfaces should also be able to be represented by this.

# So let's just say state = x \in R^n
# Then, when we have k cells, we have X \in R^{k x n}
# We have some autonomous transition dynamics X' = f(X)
# The important one for dynamics: we have b(x) = 0 or 1 if cell will divide. Binary transition
# We also need a death function: d(x) = 0 or 1 if cell will die. Binary transition
# That's actually all that's required for simulation

# But for aesthetics, we need a few more functions.
# For connectivity, we have a delta function c(x_i, x_j) = 1 if i and j are connected, 0 otherwise
# c will basically have to be 'encompassed' within f
# For surfaces, we will have a function s(x_i, x_j, x_k) = 0 or 1 if they are on the same surface. Requires all 3 connected I guess.
# Have something else for volume elements? hmm this gets complicated then...
# Dynamics can 'coincide' with connectivity but the c, s, v functions will be a separate function.

# The question is how am I going to parametrize f, b, d? c, s, and v should be relatively easy. But still representation of these will be a challenge for fluid sim.
# Well some preliminary observations:
#   - f should be limited by difference. Cells far apart should have limited influence on each other.
#   - f, b, and d should all 'depend' on connectivity.
#   - Establishing some 'marker' for connectivity will be an important challenge.
#   - dimensions corresponding to position will be constrained by physics (Newton's laws + fluid shit)
#   - Can I constrain other dimensions to just be diffusion process + nonlinearity?
#   - Well, we will get weird internal cellular logic too. I don't know how best to capture that.
#   - Can I assume no oscillatory internal cellular dynamics? Maybe? I think it could be avoided if all the oscillation comes from cell-cell interactions.
#   - Is this an artificial constraint though? No idea. 

# What I really need is a good way of 'exploring the parameter space' of functions, making tweaks, changes, etc.
# How is thaht?
# Wait but what about lagrangian for conc's?
# Let these thoughts percolate then converge on some!

# Crazy idea... What if there was some notion of 'Turing completeness' for interactions? Ie. you can reach every structure with x rules?


MORPHOGEN = 12
APOPTOSIS = 13
ADHESION = 14


# One of these numbers should represent connexin expression!
# 

w = 0.30
h = 0.01
l = 0.1
delta = 0.02

dipole_array = np.array([[l-delta, h, w, 0.9, -0.8, 0.0],
                         [l-delta, -h, w, 0.9, -0.8, 0.0],
                         [l-delta, h, -w, 0.9, -0.8, 0.0],
                         [l-delta, -h, -w, 0.9, -0.8, 0.0],
                         [-l+delta, h, w, 0.9, 0.8, 0.0],
                         [-l+delta, -h, w, 0.9, 0.8, 0.0],
                         [-l+delta, h, -w, 0.9, 0.8, 0.0],
                         [-l+delta, -h, -w, 0.9, 0.8, 0.0]])

class Organism():

    
    def __init__(self, n, init_state=None):
        if n < 6:
            raise ValueError("Cell State must have at least 6 dimensions (position and velocity).")
        elif init_state.shape[1] != n or init_state.shape[0] != 1:
            raise ValueError("Initial state must be 1 x n.")
        elif np.linalg.norm(init_state[0, 3:6]) != 0:
            raise Warning("Initial velocity should be zero.")

        self.X = init_state # X[0:3] is position, X[3:6] is velocity, X[6:] is internal state
        self.n = n
        self.m = 1
        self.t = 0.0
        self.dt = 0.1

        # Conc[0] will be mitogenic factor

        self.L = 1.0
        self.k = 3.5
        self.b = 13.0
        self.dt = 0.1

    def evolve(self):
        self.X += self.dt*self.f(self.X)
        self.X[:, 6:9] /= np.linalg.norm(self.X[:, 6:9], axis=1, keepdims=True)
        self.X[:, 0:3] -= np.mean(self.X[:, 0:3], axis=0)

        self.divide(self.X)
        self.t += 1
        print(self.t)
        print(self.m)
        return
    
    def f(self, X):
        X_prime = np.zeros((self.m, self.n))

        X_prime[:, 0:3] = X[:, 3:6]
        X_prime[:, 6:9] = np.cross(X[:, 9:12], X[:, 6:9])

        vertex_positions = X[:, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = np.linalg.norm(vertex_displacements, axis=2)
        vertex_unit_vectors = -vertex_displacements/vertex_distances[:, :, None]

        connectivity_mask = np.ones((self.m, self.m))

        sig_input = 20*(1.29*self.L - vertex_distances)
        connectivity_mask *= self.sigmoid(sig_input)# disconnect cells more than 1.1L apart
        connectivity_mask[np.diag_indices(self.m)] = 0.0

        #spring_force = self.k*(self.L - vertex_distances[:, :, None])*vertex_unit_vectors
        #h = self.sigmoid(50*(vertex_distances[:, :, None]-1.1))*(-25*(vertex_distances[:, :, None]-1.1)**2*self.sigmoid(5*(vertex_distances[:, :, None]-1.1))+25*(vertex_distances[:, :, None]-1.1)**2)
        #h = self.sigmoid(-5*(vertex_distances[:, :, None]-1.2))*self.k*(self.L - vertex_distances[:, :, None])
        #spring_force += 0.15*h*vertex_unit_vectors
        #spring_force[np.diag_indices(self.m)] = 0.0
        #spring_force *= connectivity_mask[:, :, None]


        vertex_unit_dipoles = X[:, 6:9]
        vertex_unit_dipoles_arr = vertex_unit_dipoles[None, :, :].repeat(dipole_array.shape[0], axis=0)


        default_vectors = np.array([[1.0, 0.0, 0.0]]).repeat(self.m, axis=0)
        default_vectors_arr = np.array([[1.0, 0.0, 0.0]]).repeat(dipole_array.shape[0], axis=0)

        dipole_R_arr = self.rotation_matrix_from_vectors_batch(default_vectors_arr, dipole_array[:, 3:6])
        
        vertex_unit_dipoles_arr = np.einsum('kij, kmj -> kmi', dipole_R_arr, vertex_unit_dipoles_arr)

        dipole_R = self.rotation_matrix_from_vectors_batch(default_vectors, vertex_unit_dipoles)

        
        #vertex_unit_dipoles_arr = vertex_unit_dipoles[None, :, :] + dipole_array[:, None, :]
        dipole_array_r1 = np.einsum('mij, kj -> kmi', dipole_R, dipole_array[:, 0:3])
        vertex_dipole_positions = vertex_positions[None, :, :] + dipole_array_r1

        vertex_dipole_displacements = vertex_dipole_positions[None, None, :, :, :] - vertex_dipole_positions[:, :, None, None, :]
        vertex_dipole_distances = np.linalg.norm(vertex_dipole_displacements, axis=-1)
        vertex_dipole_unit_vectors = -vertex_dipole_displacements/vertex_dipole_distances[:, :, :, :, None]

        #dipole_field_B = (3*vertex_unit_vectors*np.sum(vertex_unit_vectors * vertex_unit_dipoles[None, :, :], axis=-1, keepdims=True) - vertex_unit_dipoles[None, :, :]) /vertex_distances[:, :, None]**3
        dipole_field_B = (3*vertex_dipole_unit_vectors*np.sum(vertex_dipole_unit_vectors * vertex_unit_dipoles_arr[None, None, :, :, :], axis=-1, keepdims=True) - vertex_unit_dipoles_arr[None, None, :, :, :]) /vertex_dipole_distances[:, :, :, :, None]**3

        #dipole_torque = np.cross(vertex_unit_dipoles[:, None, :], dipole_field_B, axis=-1)
        dipole_torque = np.cross(vertex_unit_dipoles_arr[:, :, None, None, :], dipole_field_B, axis=-1)

        #dipole_torque[vertex_distances < 0.1] = 0.0

        # Calculate components of the force equation
        m2_dot_m1 = np.sum(vertex_unit_dipoles_arr[None, None, :, :, :] * vertex_unit_dipoles_arr[:, :, None, None, :], axis=-1, keepdims=True)
        m2_dot_r = np.sum(vertex_unit_dipoles_arr[None, None, :, :, :] * vertex_dipole_unit_vectors, axis=-1, keepdims=True)
        m1_dot_r = np.sum(vertex_unit_dipoles_arr[:, :, None, None, :] * vertex_dipole_unit_vectors, axis=-1, keepdims=True)

        # Force calculation using the derived formula:
        dipole_force_F = (3 * (m2_dot_m1 * vertex_dipole_unit_vectors +
                                            m2_dot_r * vertex_unit_dipoles_arr[:, :, None, None, :] -
                                            5 * m1_dot_r * m2_dot_r * vertex_dipole_unit_vectors) /
                                        (vertex_dipole_distances[:, :, :, :, None]**4))
        
        #dipole_force_F[vertex_dipole_distances < self.L] = 0.0
        mu = np.zeros_like(vertex_dipole_distances)
        mu[vertex_dipole_distances < self.L] = 0.0
        mu[vertex_dipole_distances >= self.L] = np.clip(1.0*(vertex_dipole_distances[vertex_dipole_distances >= self.L] - self.L), 0.0, 1.0)
        # Use mu to scale the component of dipole_force_F that is in the direction of the line connecting the dipoles
        # Only scale the component

        mu_component = np.zeros_like(dipole_force_F)
        mu_component = np.sum(dipole_force_F * vertex_dipole_unit_vectors, axis=-1, keepdims=True) * vertex_dipole_unit_vectors
        dipole_force_F = dipole_force_F - mu_component + mu[:, :, :, :, None]*mu_component

        dipole_torque[vertex_dipole_distances < 0.5*self.L] = 0.0
        dipole_force_F[vertex_dipole_distances < 0.5*self.L] = 0.0

        torque_from_dipole_force = np.cross(dipole_array_r1[:, :, None, None, :], dipole_force_F, axis=-1)
        torque_from_dipole_force = np.sum(torque_from_dipole_force, axis=(0, 2))
        torque_from_dipole_force[np.diag_indices(self.m)] = 0.0
        torque_from_dipole_force = np.sum(torque_from_dipole_force, axis=1)

        dipole_force_F = np.sum(dipole_force_F, axis=(0, 2))
        dipole_torque = np.sum(dipole_torque, axis=(0, 2))

        # Must calculate displaced force contribution to torque!!!!

        #pdb.set_trace()
        dipole_torque[np.diag_indices(self.m)] = 0.0
        dipole_force_F[np.diag_indices(self.m)] = 0.0

        dipole_force = np.sum(dipole_force_F, axis=1)
        dipole_torque = np.sum(dipole_torque, axis=1)

        # if self.t > 500:
        #     pdb.set_trace()

        #pdb.set_trace()

        #X_prime[:, 6:9] = np.sum(dipole_torque, axis=1)
        #pdb.set_trace()

        # if self.t > 400:
        #      pdb.set_trace()

        spring_force = self.k*(self.L - vertex_dipole_distances[:, :, :, :, None])*vertex_dipole_unit_vectors

        spring_force[vertex_dipole_distances > 1.0] = 0.0

        torque_from_spring_force = np.cross(dipole_array_r1[:, :, None, None, :], spring_force, axis=-1)
        torque_from_spring_force = np.sum(torque_from_spring_force, axis=(0, 2))
        torque_from_spring_force[np.diag_indices(self.m)] = 0.0
        torque_from_spring_force = np.sum(torque_from_spring_force, axis=1)

        spring_force = np.sum(spring_force, axis=(0, 2))
        spring_force[np.diag_indices(self.m)] = 0.0
        spring_force = np.sum(spring_force, axis=1)

        velocities = X[:, 3:6]
        #pdb.set_trace()
        # velocities_proj = np.sum(velocities[:, None, :]*vertex_unit_vectors, axis=-1)[:, :, None]*vertex_unit_vectors
        # velocities_proj[np.diag_indices(self.m)] = 0.0
        # velocities_proj *= connectivity_mask[:, :, None]
        # friction_force = -self.b*velocities_proj
        friction_force = -self.b*velocities
        #friction_force = np.sum(friction_force, axis=1)
        pressure_force = 0
        if self.t > 850:
            # Calculate pressure force - unit vector

            pressure_force = 1.2*X[:, 0:3]/np.linalg.norm(X[:, 0:3], axis=1, keepdims=True)

        X_prime[:, 3:6] = 0.18*spring_force + friction_force + 0.02*dipole_force + pressure_force

        total_torque =  0.2*dipole_torque + 0.2*torque_from_spring_force + 0.2*torque_from_dipole_force 

        # Calculate rate of change of dipole, based on torque
        X_prime[:, 9:12] = total_torque - 15.5*X[:, 9:12]

        if self.m < 4 and self.t % 20 == 0:
            divider = np.random.randint(0, self.m)
            X_prime[0, MORPHOGEN] = 1.1/self.dt
        if self.m < 24 and self.m >= 3 and self.t % 100 == 0 and self.t > 1200:
            divider = np.random.randint(0, self.m)
            X_prime[divider, MORPHOGEN] = 1.1/self.dt

        if self.t > 800 and self.t % 20 == 0:
            dipole_array[:, 4] *= 0.99
            dipole_array[:, 1] *= 1.00
            print(dipole_array[0, 4])

        #if self.t > 120:
        #    X_prime[:, ADHESION] = 0.3*(4 - np.sum(connectivity_mask > 0.5, axis=1))
        return X_prime
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def rotation_matrix_from_vectors_batch(self, default_vectors, target_vectors):
        """ Find rotation matrices that align each default vector to its corresponding target vector """
        # Normalize the vectors
        default_vectors_norm = default_vectors / np.linalg.norm(default_vectors, axis=-1, keepdims=True)
        target_vectors_norm = target_vectors / np.linalg.norm(target_vectors, axis=-1, keepdims=True)

        # Cross product of each pair
        v = np.cross(default_vectors_norm, target_vectors_norm, axis=-1)
        c = np.sum(default_vectors_norm * target_vectors_norm, axis=-1)  # Dot product for each pair
        
        # Calculate sine using the magnitude of the cross product
        s = np.linalg.norm(v, axis=-1)
        # sign of sine

        # Skew-symmetric cross-product matrices for each v
        kmat = np.zeros((v.shape[0], 3, 3))
        kmat[:, 0, 1] = -v[:, 2]
        kmat[:, 0, 2] = v[:, 1]
        kmat[:, 1, 0] = v[:, 2]
        kmat[:, 1, 2] = -v[:, 0]
        kmat[:, 2, 0] = -v[:, 1]
        kmat[:, 2, 1] = v[:, 0]

        # Rodrigues' rotation formula
        identity_mat = np.eye(3)[np.newaxis, :, :]  # Creating an identity matrix for each vector in the batch
        rotation_matrices = identity_mat + s[:, np.newaxis, np.newaxis] * kmat + \
                    (1 - c[:, np.newaxis, np.newaxis]) * (kmat @ kmat)        
        return rotation_matrices
    
    def divide(self, X):

        for i in range(self.m):
            if X[i, MORPHOGEN] > 1.0:
                self.m += 1
                new_xi = X[i] + np.random.normal(0.0, 0.01, (self.n))
                new_xi[0:3] += 0.05*X[i, 6:9]
                new_xi[3:6] += X[i, 6:9]

                new_xi[10] = X[10]/2
                new_xi[MORPHOGEN] = 0.0
                self.X[i, MORPHOGEN] = 0.0
                self.X = np.vstack((self.X, new_xi))

        return
    
    def get_vertices(self):
        return [v for v in range(self.X.shape[0])]
    
    def get_edges(self):
        vertex_positions = self.X[:, 0:3]
        vertex_displacements = vertex_positions[None, :, :] - vertex_positions[:, None, :] # (m x m x 3)
        vertex_distances = np.linalg.norm(vertex_displacements, axis=2)

        connectivity_mask = np.ones((self.m, self.m))
        sig_input = 20*(self.L - vertex_distances)
        connectivity_mask *= self.sigmoid(sig_input) # disconnect cells more than 2L apart

        edges = []
        for i in range(self.m):
            for j in range(i+1, self.m):
                if connectivity_mask[i, j] >= 0.5:
                    edges.append((i, j))
        return edges
    
    def get_layout(self):
        return {i: self.X[i, 0:3] for i in range(self.m)}