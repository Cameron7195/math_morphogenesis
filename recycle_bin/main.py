
from manim import *
from Organism import *
import numpy as np

w = 0.20
h = 0.40
l = 0.20
delta = 0

dipole_array = np.array([[l-delta, h, w, 0.9, -0.9, 0.0],
                         [l-delta, -h, w, 0.9, -0.9, 0.0],
                         [l-delta, h, -w, 0.9, -0.9, 0.0],
                         [l-delta, -h, -w, 0.9, -0.9, 0.0],
                         [-l+delta, h, w, 0.9, 0.9, 0.0],
                         [-l+delta, -h, w, 0.9, 0.9, 0.0],
                         [-l+delta, h, -w, 0.9, 0.9, 0.0],
                         [-l+delta, -h, -w, 0.9, 0.9, 0.0]])

class GraphExample(Scene):
    def construct(self):
        pos = [np.array([0.0, 0.0, 0.0])]
        org = Organism(15, np.array([0.0,# x           0
                                    0.0, # y            1
                                    0.0, # z            2
                                    0.0, # x'           3
                                    0.0, # y'           4
                                    0.0, # z'           5
                                    1.0, # dipole_x     6
                                    0.0, # dipole_y     7
                                    0.0, # dipole_z     8
                                    0.0, # dipole_x'    9
                                    0.0, # dipole_y'    10
                                    0.0, # dipole_z'    11
                                    0.98, # morphogen    12
                                    0.0, # apoptosis    13
                                    0.0] # adhesion     14
                                    ).reshape(1, 15))

        V = org.get_vertices()
        E = org.get_edges()
        layout = org.get_layout()
        
        D = Graph(
            V,
            E,
            layout=layout
        )
        self.add(D)
        #self.play(D)
        self.wait()

        for i in range(0, 400):
            #D.vertices[1].move_to([1, 1 + i/100, 0])
            org.evolve()

            V = org.get_vertices()
            E = org.get_edges()
            layout = org.get_layout()

            self.remove(D)
            D = Graph(
                V,
                E,
                layout=layout
            )

            # Draw arrow attached to each point, based on roll, pitch, yaw components
            arrows = []
            for v in V:
                #print(org.X[v, 6:9])
                end_pt = np.array(org.X[v, 6:9])
                # Convert to x, y, z point on the unit sphere
                arrow = Arrow(ORIGIN, end_pt, buff=0, stroke_width=3, max_stroke_width_to_length_ratio=2, max_tip_length_to_length_ratio=0.15)
                arrows += [arrow]
                arrow.move_to(D.vertices[v].get_center() + end_pt/2)
                self.add(arrow)

            # draw points corresponding to each element in dipole array, for each vertex
            points = []
            vertex_positions = org.X[:, 0:3]
            vertex_unit_dipoles = org.X[:, 6:9]
            default_vectors = np.array([[1.0, 0.0, 0.0]]).repeat(org.m, axis=0)
            dipole_R = org.rotation_matrix_from_vectors_batch(default_vectors, vertex_unit_dipoles)
            dipole_array_r1 = np.einsum('mij, kj -> kmi', dipole_R, dipole_array[:, 0:3])
            vertex_dipole_positions = vertex_positions[None, :, :] + dipole_array_r1
            default_vectors_arr = np.array([[1.0, 0.0, 0.0]]).repeat(dipole_array.shape[0], axis=0)

            dipole_R_arr = org.rotation_matrix_from_vectors_batch(default_vectors_arr, dipole_array[:, 3:6])
            vertex_unit_dipoles_arr = vertex_unit_dipoles[None, :, :].repeat(dipole_array.shape[0], axis=0)
            vertex_unit_dipoles_arr = np.einsum('kij, kmj -> kmi', dipole_R_arr, vertex_unit_dipoles_arr)

            for k in range(8):
                for v in range(org.m):
                    end_pt = vertex_unit_dipoles_arr[k, v, :]
                    arrow = Arrow(ORIGIN, end_pt, buff=0, stroke_width=2, max_stroke_width_to_length_ratio=0.5, max_tip_length_to_length_ratio=0.10, color=BLUE)
                    arrow.move_to(vertex_dipole_positions[k, v, :] + end_pt/2)
                    # Add small point
                    self.add(arrow)
                    points += [arrow]


            # If two vertices, color them red and blue
            # if len(V) == 2:
            #     D.vertices[0].set_color(RED)
            #     D.vertices[1].set_color(BLUE)

            self.add(D)
            self.wait(0.1)

            # Remove arrows
            for arrow in arrows:
                self.remove(arrow)
            for point in points:
                self.remove(point)
            #self.play(create(D), run_time=0.1)
            
        self.wait()
        return
    
# Write an animation for cell division. Build the class for a graph. Start some simple simulations.

