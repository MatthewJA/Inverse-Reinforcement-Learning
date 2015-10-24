"""
Implements deep maximum entropy inverse reinforcement learning based on
Ziebart et al., 2008 and Wulfmeier et al., 2015, using symbolic methods with
Theano.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn
import theano as th
import theano.tensor as T

from . import maxent

FLOAT = th.config.floatX

def find_svf(n_states, trajectories):
    """
    Find the state vistiation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return th.shared(svf, "svf", borrow=True)

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = T.zeros(n_states, dtype=FLOAT)

    def update(s, prev_diff, v, reward, tps):
        max_v = float("-inf")
        v_template = T.zeros_like(v)
        for a in range(n_actions):
            tp = tps[s, a, :]
            max_v = T.largest(max_v, T.dot(tp, reward + discount*v))
        new_diff = abs(v[s] - max_v)
        if T.lt(prev_diff, new_diff):
            diff = new_diff
        else:
            diff = prev_diff
        return (diff, T.set_subtensor(v_template[s], max_v)), {}

    def until_converged(diff, v):
        (diff, vs), _ = th.scan(
                fn=update,
                outputs_info=[{"initial": diff, "taps": [-1]},
                              None],
                sequences=[T.arange(n_states)],
                non_sequences=[v, reward, transition_probabilities])
        return ((diff[-1], vs.sum(axis=0)), {},
                th.scan_module.until(diff[-1] < threshold))

    (_, vs), _ = th.scan(fn = until_converged,
                         outputs_info=[
                            # Need to force an inf into the right Theano
                            # data type and this seems to be the only way that
                            # works.
                            {"initial": getattr(np, FLOAT)(float("inf")),
                             "taps": [-1]},
                            {"initial": v,
                             "taps": [-1]}],
                         n_steps=1000)

    return vs[-1]

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Optimal value array (if known). Default None.
    -> Action probabilities for each state.
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    # Get Q using equation 9.2 from Ziebart's thesis.
    Q = T.zeros((n_states, n_actions))
    def make_Q(i, j, tps, Q, reward, v):
        Q_template = T.zeros_like(Q)
        tp = transition_probabilities[i, j, :]
        return T.set_subtensor(Q_template[i, j], tp.dot(reward + discount*v)),{}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    Qs, _ = th.scan(fn=make_Q,
                    outputs_info=None,
                    sequences=[state_range, action_range],
                    non_sequences=[transition_probabilities, Q, reward, v])
    Q = Qs.sum(axis=0)
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = T.exp(Q)/T.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = find_policy(n_states, n_actions,
                         transition_probability, r, discount)

    start_state_count = T.extra_ops.bincount(trajectories[:, 0, 0],
                                             minlength=n_states)
    p_start_state = start_state_count.astype(FLOAT)/n_trajectories

    def state_visitation_step(i, j, prev_svf, policy, tps):
        """
        The sum of the outputs of a scan over this will be a row of the svf.
        """

        svf = prev_svf[i] * policy[i, j] * tps[i, j, :]
        return svf, {}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    def state_visitation_row(prev_svf, policy, tps, state_range, action_range):
        svf_t, _ = th.scan(fn=state_visitation_step,
                           sequences=[state_range, action_range],
                           non_sequences=[prev_svf, policy, tps])
        svf_t = svf_t.sum(axis=0)
        return svf_t, {}

    svf, _ = th.scan(fn=state_visitation_row,
                     outputs_info=[{"initial": p_start_state, "taps": [-1]}],
                     n_steps=trajectories.shape[1]-1,
                     non_sequences=[policy, transition_probability, state_range,
                                 action_range])

    return svf.sum(axis=0) + p_start_state

def irl(structure, feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, initialisation="normal", l1=0.1,
        l2=0.1):
    """
    Find the reward function for the given trajectories.

    structure: Neural network structure tuple, e.g. (10, 3, 3) would be a
        3-layer neural network with 10 inputs.
    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    initialisation: What distribution to use. str in {normal, uniform}. Default
        normal.
    l1: L1 regularisation. Default 0.1. float.
    l2: L2 regularisation. Default 0.1. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape
    transition_probability = th.shared(transition_probability, borrow=True)
    trajectories = th.shared(trajectories, borrow=True)

    # Initialise W matrices; b biases.
    n_layers = len(structure)-1
    weights = []
    hist_w_grads = []  # For AdaGrad.
    biases = []
    hist_b_grads = []  # For AdaGrad.
    for i in range(n_layers):
        # W
        shape = (structure[i+1], structure[i])
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="W", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="W", borrow=True)
        weights.append(matrix)
        hist_w_grads.append(th.shared(np.zeros(shape), name="hdW", borrow=True))

        # b
        shape = (structure[i+1], 1)
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="b", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="b", borrow=True)
        biases.append(matrix)
        hist_b_grads.append(th.shared(np.zeros(shape), name="hdb", borrow=True))

    # Initialise α weight, β bias.
    if initialisation == "normal":
        α = th.shared(rn.normal(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    else:
        α = th.shared(rn.uniform(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    hist_α_grad = T.zeros(α.shape)  # For AdaGrad.

    adagrad_epsilon = 1e-6  # AdaGrad numerical stability.

    #### Theano symbolic setup. ####

    # Symbolic input.
    s_feature_matrix = T.matrix("x")
    # Feature matrices.
    # All dimensions of the form (d_layer, n_states).
    φs = [s_feature_matrix.T]
    # Forward propagation.
    for W, b in zip(weights, biases):
        φ = T.nnet.sigmoid(th.compile.ops.Rebroadcast((0, False), (1, True))(b)
                           + W.dot(φs[-1]))
        φs.append(φ)
        # φs[1] = φ1 etc.
    # Reward.
    r = α.dot(φs[-1]).reshape((n_states,))
    # Engineering hack: z-score the reward.
    r = (r - r.mean())/r.std()
    # Associated feature expectations.
    expected_svf = find_expected_svf(n_states, r,
                                     n_actions, discount,
                                     transition_probability,
                                     trajectories)
    svf = maxent.find_svf(n_states, trajectories.get_value())
    # Derivatives (backward propagation).
    updates = []
    α_grad = φs[-1].dot(svf - expected_svf).T
    hist_α_grad += α_grad**2
    adj_α_grad = α_grad/(adagrad_epsilon + T.sqrt(hist_α_grad))
    updates.append((α, α + adj_α_grad*learning_rate))

    def grad_for_state(s, theta, svf_diff, r):
        """
        Calculate the gradient with respect to theta for one state.
        """

        regularisation = abs(theta).sum()*l1 + (theta**2).sum()*l2
        return svf_diff[s] * T.grad(r[s], theta) - regularisation, {}

    for i, W in enumerate(weights):
        w_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[W, svf - expected_svf, r])
        w_grad = w_grads.sum(axis=0)
        hist_w_grads[i] += w_grad**2
        adj_w_grad = w_grad/(adagrad_epsilon + T.sqrt(hist_w_grads[i]))
        updates.append((W, W + adj_w_grad*learning_rate))
    for i, b in enumerate(biases):
        b_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[b, svf - expected_svf, r])
        b_grad = b_grads.sum(axis=0)
        hist_b_grads[i] += b_grad**2
        adj_b_grad = b_grad/(adagrad_epsilon + T.sqrt(hist_b_grads[i]))
        updates.append((b, b + adj_b_grad*learning_rate))

    train = th.function([s_feature_matrix], updates=updates, outputs=r)
    run = th.function([s_feature_matrix], outputs=r)

    for e in range(epochs):
        reward = train(feature_matrix)

    return reward.reshape((n_states,))
