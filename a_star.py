#!/usr/bin/python

"""
BSD 2-Clause License
Copyright (c) 2017, Andrew Dahdouh
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CON   TRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.pyplot as plt
from copy import deepcopy
import math

COLOR_MAP = (0, 8)


class PathPlanner:

    def __init__(self, grid, visual=False):
        """
        Constructor of the PathPlanner Class.
        :param grid: List of lists that represents the
        occupancy map/grid. List should only contain 0's
        for open nodes and 1's for obstacles/walls.
        :param visual: Boolean to determine if Matplotlib
        animation plays while path is found.
        """
        self.grid = grid
        self.visual = visual
        self.heuristic = None
        self.goal_node = None

    def calc_heuristic(self):
        """
        Function will create a list of lists the same size
        of the occupancy map, then calculate the cost from the
        goal node to every other node on the map and update the
        class member variable self.heuristic.
        :return: None.
        """
        row = len(self.grid)
        col = len(self.grid[0])
        layer = len(self.grid[0][0])

        self.heuristic = [[[0 for x in range(col)] for y in range(row)] for z in range(layer)]
        for i in range(row):
            for j in range(col):
                for k in range(layer):
                    row_diff = abs(i - self.goal_node[0])
                    col_diff = abs(j - self.goal_node[1])
                    layer_diff = abs(k - self.goal_node[2])
                    #self.heuristic[i][j][k] = int(abs(row_diff - col_diff) + min(row_diff, col_diff) * 2)
                    self.heuristic[i][j][k] = math.sqrt(row_diff**2 + col_diff**2 + layer_diff**2)

        print("Heuristic:")
        print(self.heuristic)

    def a_star(self, start_cart, goal_cart):
        """
        A* Planner method. Finds a plan from a starting node
        to a goal node if one exits.
        :param init: Initial node in an Occupancy map. [x, y, z].
        Type: List of Ints.
        :param goal: Goal node in an Occupancy map. [x, y, z].
        Type: List of Ints.
        :return: Found path or -1 if it fails.
        """
        goal = [goal_cart[2], goal_cart[1], goal_cart[0]]
        self.goal_node = goal
        init = [start_cart[2], start_cart[1], start_cart[0]]
        # Calculate the Heuristic for the map
        self.calc_heuristic()

        print(init, goal)

        if self.visual:
            viz_map = deepcopy(self.grid)
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            ax.set_title('Occupancy Grid')
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            plt.zticks(visible=False)
            plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
            ax.set_aspect('equal')
            plt.pause(2)
            viz_map[init[0]][init[1]][init[2]] = 5  # Place Start Node
            viz_map[goal[0]][goal[1]][goal[2]] = 6
            plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
            plt.pause(2)

        # Different move/search direction options:

        #delta = [[-1, 0],  # go up
                 #[0, -1],  # go left
                 #[1, 0],  # go down
                 #[0, 1]]  # go right
        #delta_name = ['^ ', '< ', 'v ', '> ']

        # If you wish to use diagonals:
        # (z, y, x)
        delta = [[0, -1, 0],  # go up in plane
                 [0, 0, -1],  # go left in plane
                 [0, 1, 0],  # go down in plane
                 [0, 0, 1],  # go right in plane
                 [0, -1, -1],  # upper left in plane
                 [0, 1, -1],  # lower left in plane
                 [0, -1, 1],  # upper right in plane
                 [0, 1, 1],  # lower right in plane
                 [-1, -1, 0],  # go up and rise
                 [-1, 0, -1],  # go left and rise
                 [-1, 1, 0],  # go down and rise
                 [-1, 0, 1],  # go right and rise
                 [-1, -1, -1],  # upper left and rise
                 [-1, 1, -1],  # lower left and rise
                 [-1, -1, 1],  # upper right and rise
                 [-1, 1, 1],  # lower right and rise
                 [1, -1, 0],  # go up and sink
                 [1, 0, -1],  # go left and sink
                 [1, 1, 0],  # go down and sink
                 [1, 0, 1],  # go right and sink
                 [1, -1, -1],  # upper left and sink
                 [1, 1, -1],  # lower left and sink
                 [1, -1, 1],  # upper right and sink
                 [1, 1, 1]]  # lower right and sink
        delta_name = ['^0 ', '<0 ', 'v0 ', '>0 ', 'UL0', 'LL0', 'UR0', 'LR0', 
                    '^F ', '<F ', 'vF ', '>F ', 'ULF', 'LLF', 'URF', 'LRF', 
                    '^S ', '<S ', 'vS ', '>S ', 'ULS', 'LLS', 'URS', 'LRS'] 
                    #F is float (rise), S is sink, 0 is stationary on z axis

        # Heavily used from some of the A* Examples by Sebastian Thrun:

        closed = [[[0 for col in range(len(self.grid[0]))] for row in range(len(self.grid))] for layer in range(len(self.grid[0][0]))]
        shortest_path = [[['  ' for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))] for _ in range(len(self.grid[0][0]))]
        closed[init[0]][init[1]][init[2]] = 1

        expand = [[[-1 for col in range(len(self.grid[0]))] for row in range(len(self.grid))] for layer in range(len(self.grid[0][0]))]
        delta_tracker = [[[-1 for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))] for _ in range(len(self.grid[0][0]))]

        cost = 1
        x = init[0]
        y = init[1]
        z = init[2]
        g = 0
        f = g + self.heuristic[x][y][z]
        open = [[f, g, x, y, z]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand
        count = 0
        while not found and not resign:
            if len(open) == 0:
                resign = True
                if self.visual:
                    plt.text(2, 10, s="No path found...", fontsize=18, style='oblique', ha='center', va='top')
                    plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
                    plt.pause(5)
                return -1
            else:
                open.sort()
                open.reverse()
                next = open.pop()
                x = next[2]
                y = next[3]
                z = next[4]
                g = next[1]
                expand[x][y][z] = count
                count += 1

                if x == goal[0] and y == goal[1] and z == goal[2]:
                    found = True
                    if self.visual:
                        viz_map[goal[0]][goal[1]][goal[2]] = 7
                        plt.text(2, 10, s="Goal found!", fontsize=18, style='oblique', ha='center', va='top')
                        plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
                        plt.pause(2)
                else:
                    for i in range(len(delta)):
                        x2 = x + delta[i][0]
                        y2 = y + delta[i][1]
                        z2 = z + delta[i][2]
                        if len(self.grid) > x2 >= 0 and len(self.grid[0]) > y2 >= 0 and len(self.grid[0][0]) > z2 >= 0:
                            if closed[x2][y2][z2] == 0 and self.grid[x2][y2][z2] == 0:
                                g2 = g + cost
                                f = g2 + self.heuristic[x2][y2][z2]
                                open.append([f, g2, x2, y2, z2])
                                closed[x2][y2][z2] = 1
                                delta_tracker[x2][y2][z2] = i
                                if self.visual:
                                    viz_map[x2][y2][z2] = 3
                                    plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
                                    plt.pause(.5)

        current_x = goal[0]
        current_y = goal[1]
        current_z = goal[2]
        shortest_path[current_x][current_y][current_z] = '* '
        full_path = []
        while current_x != init[0] or current_y != init[1] or current_z != init[2]:
            previous_x = current_x - delta[delta_tracker[current_x][current_y][current_z]][0]
            previous_y = current_y - delta[delta_tracker[current_x][current_y][current_z]][1]
            previous_z = current_z - delta[delta_tracker[current_x][current_y][current_z]][2]
            shortest_path[previous_x][previous_y][previous_z] = delta_name[delta_tracker[current_x][current_y][current_z]]
            full_path.append((current_x, current_y, current_z))
            current_x = previous_x
            current_y = previous_y
            current_z = previous_z
        full_path.reverse()
        print("Found the goal in {} iterations.".format(count))
        print("full_path: ", full_path[:-1])
        print("shortest path:", shortest_path)

        if self.visual:
            for node in full_path:
                viz_map[node[0]][node[1]][node[2]] = 7
                plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
                plt.pause(.5)

            # Animate reaching goal:
            viz_map[goal[0]][goal[1]][node[2]] = 8
            plt.imshow(viz_map, origin='upper', interpolation='none', clim=COLOR_MAP)
            plt.pause(5)

        return init, full_path[:-1]


if __name__ == '__main__':
    test_grid = [[[0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0]],
                 [[0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0]],
                 [[0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0]]]
    test_start = [0, 0, 0]  # [x, y]
    test_goal = [1, 5, 7]   # [x, y]

# test_grid = [[0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0, 0, 0, 0],
#              [1, 1, 1, 1, 1, 1, 1, 1],
#              [1, 0, 0, 1, 1, 0, 0, 1],
#              [1, 0, 0, 1, 1, 0, 0, 1],
#              [1, 0, 0, 1, 1, 0, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 1],
#              [1, 1, 1, 1, 1, 1, 1, 1]]

    # test_start = [2, 4]  #  [x, y]
    # test_goal =  [6, 11]  # [x, y]

    # Create an instance of the PathPlanner class:
    test_planner = PathPlanner(test_grid, True)

    # Plan a path.
    test_planner.a_star(test_start, test_goal)