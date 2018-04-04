import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import os, sys, contextlib
from itertools import product

print("--> Loading parameters...")

global par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'load_previous_model'   : False,
    'debug_mode'            : False,

    # Network shape and configuration
    'n_hidden'              : 120,
    'n_dendrites'           : 1,
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 0.5,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.2,
    'noise_rnn_sd'          : 0.5,

    # Cost parameters
    'spike_cost'            : 1e-3,
    'wiring_cost'           : 1e-3, #1e-6,

    # Training specs
    'batch_train_size'      : 16,      # The number of Gym environments being run simultaneously
    'num_iterations'        : 200,
    'iters_between_outputs' : 20,

    # Task specs
    'environment_type'      : 'Pong-v0', #'CartPole-v0', 'Pendulum-v0'
    'num_steps'             : 20,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',
}



"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    #print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_dependencies()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']
    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    if par['EI']:
        par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    # The time step in seconds
    par['dt_sec'] = par['dt']/1000

    #############################################
    ### Setting up the OpenAI Gym environment ###
    #############################################

    # Create a sample environment
    with contextlib.redirect_stdout(None):
        sample_env = gym.make(par['environment_type'])

    par['envtype'] = np.dtype(type(sample_env))
    sample_env.reset()
    obs, _, _, _ = sample_env.step(sample_env.action_space.sample())

    # Check observation shape and action set
    par['observation_shape'] = np.shape(obs)

    # Determine action space setup
    if type(sample_env.action_space) == gym.spaces.Box:
        par['action_type'] = 'continuum'
        par['action_shape'] = sample_env.action_space.shape
        par['action_set'] = (sample_env.action_space.low, sample_env.action_space.high)

        # Translate to inputs and outputs
        par['n_output'] = par['action_shape'][0]
        par['n_input'] = par['observation_shape'][0]

    elif type(sample_env.action_space) == gym.spaces.Discrete:
        par['action_type'] = 'discrete'
        par['action_shape'] = sample_env.action_space.n
        par['action_set'] = np.arange(par['action_shape'])

        # Translate to inputs and outputs
        par['n_output'] = par['action_shape']
        par['n_input'] = par['observation_shape'][0]

    if par['debug_mode']:
        print('Possible Actions:')
        print(' | '.join(['{} : {}'.format(n, a) for n, a in \
            zip(par['action_set'], sample_env.unwrapped.get_action_meanings())]))

    ###########################################################
    ### Setting up intial weights, biases, and other values ###
    ###########################################################

    par['h_init'] = 0.1*np.ones((par['batch_train_size'], par['n_hidden']), dtype=np.float32)


update_dependencies()

print("--> Parameters successfully loaded.\n")
