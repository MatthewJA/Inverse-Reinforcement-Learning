# Inverse Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.555999.svg)](https://doi.org/10.5281/zenodo.555999)

Implements selected inverse reinforcement learning (IRL) algorithms as part of COMP3710, supervised by Dr Mayank Daswani and Dr Marcus Hutter. My final report is available [here](http://matthewja.com/pdfs/irl.pdf) and describes the implemented algorithms.

If you use this code in your work, you can cite it as follows:
```bibtex
@misc{alger16,
  author       = {Matthew Alger},
  title        = {Inverse Reinforcement Learning},
  year         = 2016,
  doi          = {10.5281/zenodo.555999},
  url          = {https://doi.org/10.5281/zenodo.555999}
}
```

## Algorithms implemented

- Linear programming IRL. From Ng & Russell, 2000. Small state space and large state space linear programming IRL.
- Maximum entropy IRL. From Ziebart et al., 2008.
- Deep maximum entropy IRL. From Wulfmeier et al., 2015; original derivation.

Additionally, the following MDP domains are implemented:
- Gridworld (Sutton, 1998)
- Objectworld (Levine et al., 2011)

## Requirements
- NumPy
- SciPy
- CVXOPT
- Theano
- MatPlotLib (for examples)

## Module documentation

Following is a brief list of functions and classes exported by modules. Full documentation is included in the docstrings of each function or class; only functions and classes intended for use outside the module are documented here.

### linear_irl

Implements linear programming inverse reinforcement learning (Ng & Russell, 2000).

**Functions:**

- `irl(n_states, n_actions, transition_probability, policy, discount, Rmax, l1)`: Find a reward function with inverse RL.
- `large_inverseRL(value, transition_probability, feature_matrix, n_states, n_actions, policy)`: Find the reward in a large state space.

### maxent
    
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008).

**Functions:**

- `irl(feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate)`: Find the reward function for the given trajectories.
- `find_svf(feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate)`: Find the state visitation frequency from trajectories.
- `find_feature_expectations(feature_matrix, trajectories)`:  Find the feature expectations for the given trajectories. This is the average path feature vector.
- `find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories)`: Find the expected state visitation frequencies using algorithm 1 from Ziebart et al. 2008.
- `expected_value_difference(n_states, n_actions, transition_probability, reward, discount, p_start_state, optimal_value, true_reward)`: Calculate the expected value difference, which is a proxy to how good a recovered reward function is.

### deep_maxent

Implements deep maximum entropy inverse reinforcement learning based on Ziebart et al., 2008 and Wulfmeier et al., 2015, using symbolic methods with Theano.

**Functions:**

- `irl(structure, feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate, initialisation="normal", l1=0.1, l2=0.1)`: Find the reward function for the given trajectories.
- `find_svf(n_states, trajectories)`: Find the state vistiation frequency from trajectories.
- `find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories)`: Find the expected state visitation frequencies using algorithm 1 from Ziebart et al. 2008.

### value_iteration

Find the value function associated with a policy. Based on Sutton & Barto, 1998.

**Functions:**

- `value(policy, n_states, transition_probabilities, reward, discount, threshold=1e-2)`: Find the value function associated with a policy.
- `optimal_value(n_states, n_actions, transition_probabilities, reward, discount, threshold=1e-2)`: Find the optimal value function.
- `find_policy(n_states, n_actions, transition_probabilities, reward, discount, threshold=1e-2, v=None, stochastic=True)`: Find the optimal policy.

### mdp

#### gridworld

Implements the gridworld MDP.

**Classes, instance attributes, methods:**

- `Gridworld(grid_size, wind, discount)`: Gridworld MDP.
    - `actions`: Tuple of (dx, dy) actions.
    - `n_actions`: Number of actions. int.
    - `n_states`: Number of states. int.
    - `grid_size`: Size of grid. int.
    - `wind`: Chance of moving randomly. float.
    - `discount`: MDP discount factor. float.
    - `transition_probability`: NumPy array with shape (n_states, n_actions, n_states) where `transition_probability[si, a, sk]` is the probability of transitioning from state si to state sk under action a.
    - `feature_vector(i, feature_map="ident")`: Get the feature vector associated with a state integer.
    - `feature_matrix(feature_map="ident")`: Get the feature matrix for this gridworld.
    - `int_to_point(i)`: Convert a state int into the corresponding coordinate.
    - `point_to_int(p)`: Convert a coordinate into the corresponding state int.
    - `neighbouring(i, k)`: Get whether two points neighbour each other. Also returns true if they are the same point.
    - `reward(state_int)`: Reward for being in state state_int.
    - `average_reward(n_trajectories, trajectory_length, policy)`: Calculate the average total reward obtained by following a given policy over n_paths paths.
    - `optimal_policy(state_int)`: The optimal policy for this gridworld.
    - `optimal_policy_deterministic(state_int)`: Deterministic version of the optimal policy for this gridworld.
    - `generate_trajectories(n_trajectories, trajectory_length, policy, random_start=False)`: Generate n_trajectories trajectories with length trajectory_length, following the given policy.

#### objectworld

Implements the objectworld MDP described in Levine et al. 2011.

**Classes, instance attributes, methods:**

- `OWObject(inner_colour, outer_colour)`: Object in objectworld.
    - `inner_colour`: Inner colour of object. int.
    - `outer_colour`: Outer colour of object. int.

- `Objectworld(grid_size, n_objects, n_colours, wind, discount)`: Objectworld MDP.
    - `actions`: Tuple of (dx, dy) actions.
    - `n_actions`: Number of actions. int.
    - `n_states`: Number of states. int.
    - `grid_size`: Size of grid. int.
    - `n_objects`: Number of objects in the world. int.
    - `n_colours`: Number of colours to colour objects with. int.
    - `wind`: Chance of moving randomly. float.
    - `discount`: MDP discount factor. float.
    - `objects`: Set of objects in the world.
    - `transition_probability`: NumPy array with shape (n_states, n_actions, n_states) where `transition_probability[si, a, sk]` is the probability of transitioning from state si to state sk under action a.
    - `feature_vector(i, discrete=True)`: Get the feature vector associated with a state integer.
    - `feature_matrix(discrete=True)`: Get the feature matrix for this gridworld.
    - `int_to_point(i)`: Convert a state int into the corresponding coordinate.
    - `point_to_int(p)`: Convert a coordinate into the corresponding state int.
    - `neighbouring(i, k)`: Get whether two points neighbour each other. Also returns true if they are the same point.
    - `reward(state_int)`: Reward for being in state state_int.
    - `average_reward(n_trajectories, trajectory_length, policy)`: Calculate the average total reward obtained by following a given policy over n_paths paths.
    - `generate_trajectories(n_trajectories, trajectory_length, policy)`: Generate n_trajectories trajectories with length trajectory_length, following the given policy.
