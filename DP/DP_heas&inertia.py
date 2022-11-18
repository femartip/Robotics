import subprocess
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import rospy as ros
from rospy import Publisher
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String

BLUE = "#1E64C8"
PINK = "#E85E71"
RED = "#DC4E28"
BLACK = "#000000"

SR = 100  # "Star-Reward" which we will give when arriving on target
# Motion-Reward Table that describes the map in the picture:
# > I chose to double the array size so that we can describe the thin walls as if they were
#   intermediate cells.
# > I have included the outer walls too.
# > code for walls: -1 means obstacle, 0 means we can pass
# > code for cells: 0,1,2 -> height of ground (white, light grey, dark grey respectively); I add
#   +SR on cell with a reward (so 101 would be light grey with reward)
GRID = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
                 [-1., +0., +0., +0., +0., +0., +0., +0., -1., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., +0., +0., -1., -1., -1., +0., -1., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., +0., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., +0., +0., -1., -1., -1., +0., -1., -1., -1., -1., -1., -1., -1.],
                 [-1., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +SR, -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., -1., -1., -1., -1., -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., +0., -1.],
                 [-1., +1., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., -1., -1., -1., +1., +0., +0., +0., +0., +0., +0., +0., +1., -1.],
                 [-1., +0., -1., -1., -1., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., +0., +1., +0., +2., +0., +1., +0., +0., +0., +1., +0., +2., -1.],
                 [-1., +0., +0., +0., +0., +0., +0., +0., -1., -1., -1., -1., -1., +0., -1.],
                 [-1., +0., +0., +0., +0., +1., +0., +0., +0., +0., +0., +1., +0., +2., -1.],
                 [-1., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., -1.],
                 [-1., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +0., +1., -1.],
                 [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])


class DPstandard:
    QUEUE_SIZE = 2

    def __init__(self, name: str = "dp_navigation_planner"):
        self.node_name: str = name

        # Declare ROS publisher names
        self.sys_pub_name: str = "/syscommand"
        self.sys_pub: Union[None, Publisher] = None
        self.nav_pub_name: str = "/nav_planner"
        self.nav_pub: Union[None, Publisher] = None

    def start_ros(self):
        """
        Initialize ros and set all Subscribers and Publishers
        :return:
        """
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.nav_pub = ros.Publisher(self.nav_pub_name, numpy_msg(Floats),
                                     queue_size=self.QUEUE_SIZE)
        self.sys_pub = ros.Publisher(self.sys_pub_name, String, queue_size=self.QUEUE_SIZE)
        time.sleep(1)

    def run(self):
        self.__reset_simulation()

        t0 = time.time()
        V, U = self.compute_value_function_and_policy()
        print(f"Solution found in {(time.time() - t0) * 1000:.0f} milliseconds.")

        self.__publish_policy(U)

        self.__visualize(V, U)

    def motion_rewards_table(self, row, column, heading):
        """
        The function which computes the motions & rewards at each cell.

        You could also decide to write a list where, at ach position, you list the possible actions
        and associated results. This would require less preparatory thinking, but much more tedious
        writing, especially for big mazes. You can try it out if you want.

        Here, we write the code which "implements the rules on the picture". This requires a bit of
        thinking, but afterwards the maze-dependent part reduces to just encoding the picture.
        :param row:
        :param column:
        :param heading: 0: East - 1: North - 2: West - 3: South - 4: not-moved-yet
        :return:
        """

        options = []

        # given cell indices, compute where they are in the Motion-Reward Table (since it has been
        # stretched out to accomodate the thin walls). Keep in mind that numpy reads arrays
        # left-to-right and top-to-bottom -- so North is in the negative direction.
        y, x = row * 2 + 1, column * 2 + 1

        if GRID[y, x] == -1:  # for the sake of convenience: if in an obstacle, stay there
            options.append({"arrival_state": (row, column, heading), "reward": 0})
        if GRID[y, x] >= SR:  # Special case of reward cell: game ends (stay there)
            options.append({"arrival_state": (row, column, heading), "reward": 0})
        else:
            # Check height of current cell (the % SR is only there to retrieve the height of the
            # goal state - e.g. 101 % 100 = 1 - if we had defined our motion reward table in a less
            # lazy manner, this would not be necessary)
            current_height = GRID[y, x] % SR

            # YOUR SOLUTION:
            # 1. Compute previous height. If the "previous" square is an unreachable location, we
            #    can assign an arbitrary value, because the backwards algorithm can never get
            #    there. We just need to make sure that the program does not crash in those cases.
            try:
                if heading == 0:  # East
                    previous_height = GRID[y, x - 2] % SR
                elif heading == 1:  # North
                    previous_height = GRID[y + 2, x] % SR
                elif heading == 2:  # West
                    previous_height = GRID[y, x + 2] % SR
                elif heading == 3:  # South
                    previous_height = GRID[y - 2, x] % SR
                else:
                    previous_height = current_height
            except IndexError:
                previous_height = current_height
            # END EDIT

            # East motion
            if not GRID[y, x + 1] < 0:  # cannot move east if there is a wall there
                future_height = GRID[y, x + 2] % SR  # height of cell we would move to
                # check if this moves gives us a big reward (star), or else this is just 0
                reward = SR if GRID[y, x + 2] >= SR else 0

                # YOUR SOLUTION:
                # 2. checking the effect of previous motion direction. We suggest taht you compute
                #    "headrew" the heading reward here. Of course it will depend on "heading", the
                #    previous motion direction. You may have to check if we are inside valid grid
                #    cells. Then just add "headrew" to the previous rewards that we found in
                #    exercise 1.
                if future_height == current_height:
                    reward -= 1
                elif future_height > current_height:
                    reward -= 2
                elif future_height < current_height:
                    reward -= 0.5

                if heading == 0:
                    reward += 0
                else:       #Costs of turning
                    if current_height == previous_height:
                        reward -= 1
                    elif current_height > previous_height:
                        reward -= 0.2
                    elif current_height < previous_height:
                        reward -= 2

                # END EDIT

                options.append({"arrival_state": (row, column + 1, 0), "reward": reward})

            # YOUR SOLUTION:
            # 3. Do the same for West, North, South
            
            # West motion
            if not GRID[y, x - 1] < 0:  # cannot move east if there is a wall there
                future_height = GRID[y, x - 2] % SR  # height of cell we would move to
                # check if this moves gives us a big reward (star), or else this is just 0
                reward = SR if GRID[y, x - 2] >= SR else 0

                # YOUR SOLUTION:
                # 2. checking the effect of previous motion direction. We suggest taht you compute
                #    "headrew" the heading reward here. Of course it will depend on "heading", the
                #    previous motion direction. You may have to check if we are inside valid grid
                #    cells. Then just add "headrew" to the previous rewards that we found in
                #    exercise 1.
                if future_height == current_height:
                    reward -= 1
                elif future_height > current_height:
                    reward -= 2
                elif future_height < current_height:
                    reward -= 0.5

                if heading == 2:
                    reward += 0
                else:       #Costs of turning
                    if current_height == previous_height:
                        reward -= 1
                    elif current_height > previous_height:
                        reward -= 0.2
                    elif current_height < previous_height:
                        reward -= 2
                
                options.append({"arrival_state": (row, column - 1, 2), "reward": reward})


            # North motion
            if not GRID[y - 1, x] < 0:  # cannot move east if there is a wall there
                future_height = GRID[y - 2, x] % SR  # height of cell we would move to
                # check if this moves gives us a big reward (star), or else this is just 0
                reward = SR if GRID[y - 2, x] >= SR else 0

                # YOUR SOLUTION:
                # 2. checking the effect of previous motion direction. We suggest taht you compute
                #    "headrew" the heading reward here. Of course it will depend on "heading", the
                #    previous motion direction. You may have to check if we are inside valid grid
                #    cells. Then just add "headrew" to the previous rewards that we found in
                #    exercise 1.
                if future_height == current_height:
                    reward -= 1
                elif future_height > current_height:
                    reward -= 2
                elif future_height < current_height:
                    reward -= 0.5

                if heading == 1:
                    reward += 0
                else:       #Costs of turning
                    if current_height == previous_height:
                        reward -= 1
                    elif current_height > previous_height:
                        reward -= 0.2
                    elif current_height < previous_height:
                        reward -= 2

                options.append({"arrival_state": (row-1, column, 1), "reward": reward})

            # South motion
            if not GRID[y + 1, x] < 0:  # cannot move east if there is a wall there
                future_height = GRID[y + 2, x] % SR  # height of cell we would move to
                # check if this moves gives us a big reward (star), or else this is just 0
                reward = SR if GRID[y + 2, x] >= SR else 0

                # YOUR SOLUTION:
                # 2. checking the effect of previous motion direction. We suggest taht you compute
                #    "headrew" the heading reward here. Of course it will depend on "heading", the
                #    previous motion direction. You may have to check if we are inside valid grid
                #    cells. Then just add "headrew" to the previous rewards that we found in
                #    exercise 1.
                if future_height == current_height:
                    reward -= 1
                elif future_height > current_height:
                    reward -= 2
                elif future_height < current_height:
                    reward -= 0.5

                if heading == 3:
                    reward += 0
                else:       #Costs of turning
                    if current_height == previous_height:
                        reward -= 1
                    elif current_height > previous_height:
                        reward -= 0.2
                    elif current_height < previous_height:
                        reward -= 2

                options.append({"arrival_state": (row+1, column , 3), "reward": reward})

        return options

    def update_value_function(self, V_old):
        """
        expected input:  Value function at previous iteration. For instance, in slides notation:
                         V_{N-2:N} (best for "2 steps remaining") if you are computing V{N-3:N}
                         (best for "3 steps remaining")
        output produced: Value function at next iteration (e.g. best for "3 steps remaining"), and
                         optimal policy, U, i.e. best action to take at every cell (if "3 steps
                         remaining)

        Expected structure needed for the visualization tool:
        (See run(...) for an example where these are already initialized.)
        1. V_old and V_new are arrays of real numbers, element (j, i) contains the value of cell
           column=i (start left), and row=j (start top) (PS: this is just natural matrix order)
        2. U is an array, in which each component is a *pair* of indices (k, l). Component (j, i)
           thus contains a vector of 2 numbers (k, l), which are the indices of the cell towards
           which the robot should go when it is at (j, i).
        """

        # HERE YOUR DP SOLVER CODE PART 1
        V_new = V_old.copy()
        U = np.empty(V_old.shape + (3,))

        for r in range(V_old.shape[0]):
            for c in range(V_old.shape[1]):
                for h in range(V_old.shape[2]):       ###Now it is a 3D matrix, where heading is tajen into consideration
                    m_options = self.motion_rewards_table(r,c,h)        ##Now it is a 3D matrix, where heading is tajen into consideration
                    for option in m_options:
                        option["value"] = V_old[option["arrival_state"]] + option["reward"]
                    maximum = max(m_options, key=lambda option: option["value"])
                    V_new[r,c,h] = maximum["value"]         ##Now it is a 3D matrix, where heading is tajen into consideration
                    U[r,c,h] = np.array(maximum["arrival_state"])     ##Now it is a 3D matrix, where heading is tajen into consideration
        return V_new, U

    def compute_value_function_and_policy(self):
        # Your DP program, part 2

        # Here I show you the coded structure used by the visualization tool, for V (V-function)
        # and for U (policy)

        # We must tell how big is our state space over which we must try to move
        nr_of_columns, nr_of_rows, nr_of_headings = 7, 10, 5

        # V will contain the value function. You may want to initialize with a different value
        # than 0
        V = np.full((nr_of_rows, nr_of_columns, nr_of_headings), 0)

        # U will contain the final plan (policy)
        # > initialization is not really necessary, we do it here to show you its structure
        U = np.stack(
            np.meshgrid(np.arange(nr_of_columns),
                        np.arange(nr_of_rows),
                        np.arange(nr_of_headings)), axis=-1)[..., [1, 0, 2]]

        # HERE YOUR DP SOLVER CODE PART 2
        V_tol = 0.1
        t_max = 100

        for n in range(t_max):
            if ros.is_shutdown():
                break
            prev_V= V.copy()
            V, U = self.update_value_function(prev_V)
            print("Solution on iteration {} = {}".format(n, np.sum(V - prev_V)))
            if np.sum(V - prev_V) < V_tol: break
        
        return V, U

    def __reset_simulation(self):
        self.sys_pub.publish("reset")

    def __publish_policy(self, U):
        self.nav_pub.publish(U.reshape(-1).astype(np.float32))

    def __visualize(self, V, U):
        """
        visualize value function and policy for 2D grid

        NO NEED TO CHANGE ANYTHING
        :return:
        """
        fig = plt.figure(figsize=(14, 10))
        for ii, heading in enumerate(["East", "North", "West", "South", "not-moved-yet"]):
            plt.subplot(2, 3, ii + 1)
            plt.title(heading)
            tiles = GRID[1::2, 1::2]
            v = V[..., ii].copy().astype(float)
            m = (SR > tiles) & (tiles >= 0)
            v[m] = (v[m] - np.min(v[m])) / (np.max(v[m]) - np.min(v[m]))
            v[~m] = np.nan
            v[tiles >= SR] = 1.

            cmap = cm.get_cmap('summer')
            cmap.set_bad(BLUE)
            fig = plt.imshow(v, cmap='summer')
            cbar = plt.colorbar(fig, ticks=[0, 1])
            cbar.ax.set_yticklabels([np.min(V[m]), np.max(V[m] + 1)])

            # Add walls
            horizontal_walls = GRID[0::2, 1::2]
            for row in range(horizontal_walls.shape[0]):
                for col in range(horizontal_walls.shape[1]):
                    if horizontal_walls[row, col] == -1:
                        plt.plot([col - .5, col + .5], [row - .5, row - .5], c=BLUE, linewidth=3)
            vertical_walls = GRID[1::2, 0::2]
            for row in range(vertical_walls.shape[0]):
                for col in range(vertical_walls.shape[1]):
                    if vertical_walls[row, col] == -1:
                        plt.plot([col - .5, col - .5], [row - .5, row + .5], c=BLUE, linewidth=3)

            # Draw policy
            for row in range(U.shape[0]):
                for col in range(U.shape[1]):
                    if np.all(U[row, col, ii] == np.array([row, col, ii])):
                        continue
                    dy, dx, _ = np.split(U[row, col, ii] - np.array([row, col, ii]), 3)
                    dx, dy = dx.squeeze(), dy.squeeze()

                    x = col - 0.11 * dx  # + 0.33 * dx
                    y = row - 0.11 * dy  # + 0.33 * dy
                    dx = 0.11 * dx
                    dy = 0.11 * dy

                    plt.arrow(x, y, dx, dy, color=BLACK, width=0.05)

        plt.show()


def main():
    dp = DPstandard()
    dp.start_ros()
    dp.run()


if __name__ == '__main__':
    main()
