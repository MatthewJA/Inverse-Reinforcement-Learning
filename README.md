# Inverse Reinforcement Learning

Implements selected inverse reinforcement learning (IRL) algorithms as part of COMP3710.

## Algorithms implemented

- Linear programming IRL
    From Ng & Russell, 2000. Small state space and large state space linear programming IRL.
- Maximum entropy IRL
    From Ziebart et al., 2008.
- Deep maximum entropy IRL
    From Wulfmeier et al., 2015; original derivation.

Additionally, the following MDP domains are implemented:
- Gridworld (Sutton, 1998)
- Objectworld (Levine et al., 2011)

## Requirements
- NumPy
- SciPy
- CVXOPT
- Theano
- MatPlotLib (for examples)