import numpy as np



yaw_rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
                        [np.sin(alpha), np.cos(alpha), 0.0],
                        [0.0, 0.0, 1.0]])
pitch_rotation = np.array([[np.cos(beta), 0.0, np.sin(beta)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(beta), 0.0, np.cos(beta)]])
roll_rotation = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(gamma), -np.sin(gamma)],
                        [0.0, np.sin(gamma), np.cos(gamma)]])