# Inverse Reinforcement Learning

Implements selected inverse reinforcement learning (IRL) algorithms as part of COMP3710.

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

### maxent
    
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008).

**Functions:**

- `irl(feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate)`: Find the reward function for the given trajectories.
- `find_svf(feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate)`: Find the state visitation frequency from trajectories.
- `find_feature_expectations(feature_matrix, trajectories)`:  Find the feature expectations for the given trajectories. This is the average path feature vector.
- `find_expected_svf(n_states, r, n_actions, discount, transition_probability, trajectories)`: Find the expected state visitation frequencies using algorithm 1 from Ziebart et al. 2008.
- `expected_value_difference(n_states, n_actions, transition_probability, reward, discount, p_start_state, optimal_value, true_reward)`: Calculate the expected value difference, which is a proxy to how good a recovered reward function is.

### value_iteration

### mdp

#### gridworld

#### objectworld