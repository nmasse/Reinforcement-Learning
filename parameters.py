import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import os, sys, contextlib
from itertools import product

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './savedir/CL4/',
    'save_analysis'         : False,
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : False,
    'stabilization'         : 'pathint',
    'no_gpu'                : False,

    # Network configuration
    'synapse_config'        : None, # Full is 'std_stf'
    'exc_inh_prop'          : 1.0,       # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 20,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 120,
    'n_dendrites'           : 1,

    # Euclidean shape
    'num_sublayers'         : 2,
    'neuron_dx'             : 1.0,
    'neuron_dy'             : 1.0,
    'neuron_dz'             : 10.0,

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

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.,        # magnitude scaling factor for von Mises
    'kappa'                 : 2.0,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-3,
    'wiring_cost'           : 1e-3, #1e-6,
    'latent_cost'           : 1e-4,

    # Synaptic plasticity specs
    'tau_fast'              : 100,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 256,      # The number of Gym environments being run simultaneously
    'num_iterations'        : 200,
    'iters_between_outputs' : 20,

    # Task specs
    'environment_type'      : 'CartPole-v0', #'CartPole-v0', 'Pendulum-v0'
    'num_steps'             : 20,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 100,
    'simulation_reps'       : 100,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : True,
    'analyze_tuning'        : True,

    # Omega parameters
    'omega_c'               : 0.0,
    'omega_xi'              : 0.1,
    'last_layer_mult'       : 2,
    'scale_factor'          : 1,

    # Projection of top-down activity
    'neuron_gate_pct'       : 0.0,
    'dendrite_gate_pct'     : 0.0,
    'dynamic_topdown'       : False,
    'num_tasks'             : 12,
    'td_cost'               : 0.0,

    # Fisher information parameters
    'EWC_fisher_calc_batch' : 8, # batch size when calculating EWC
    'EWC_fisher_num_batches': 256, # number of batches size when calculating EWC
}



"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    #print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_trial_params()
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
    #par['alpha_neuron'] = 1
    #print('Setting alpha_neuron to 1')
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


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    

def initialize(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    #w = np.random.uniform(low=0, high=0.5, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):
    if A.ndim == 3:
        return np.max(abs(np.linalg.eigvals(np.sum(A, axis=1))))
    else:
        return np.max(abs(np.linalg.eigvals(np.sum(A))))


def square_locs(num_locs, d1, d2):

    locs_per_side = np.int32(np.sqrt(num_locs))
    while locs_per_side**2 < num_locs:
        locs_per_side += 1

    x_set = np.repeat(d1*np.arange(locs_per_side)[:,np.newaxis], locs_per_side, axis=1).flatten()
    y_set = np.repeat(d2*np.arange(locs_per_side)[np.newaxis,:], locs_per_side, axis=0).flatten()
    locs  = np.stack([x_set, y_set])[:,:num_locs]

    locs[0,:] -= np.max(locs[0,:])/2
    locs[1,:] -= np.max(locs[1,:])/2

    return locs

update_dependencies()

print("--> Parameters successfully loaded.\n")
