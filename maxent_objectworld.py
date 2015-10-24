"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.objectworld as objectworld
from irl.value_iteration import find_policy

def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
         learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the objectworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    wind = 0.3
    trajectory_length = 8

    ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
                                 discount)
    ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                         ground_r, ow.discount, stochastic=False)
    trajectories = ow.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s])
    feature_matrix = ow.feature_matrix()
    r = maxent.irl(feature_matrix, ow.n_actions, discount,
        ow.transition_probability, trajectories, epochs, learning_rate)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(10, 0.9, 15, 2, 20, 200, 0.01)
