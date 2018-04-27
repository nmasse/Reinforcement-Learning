import numpy as np
import tensorflow as tf
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

    # Network parameters
    'n_hidden'              : [200, 200],
    'learning_rate'         : 5e-4,
    'noise_std'             : 5e-3,

    # Reinforcement learning parameters
    'gamma'                 : 0.75,
    'epsilon'               : 0.1,

    # Cost parameters
    'spike_cost'            : 0.,

    # Training specs
    'num_epochs'            : 1000,
    'max_steps'             : 99,
    'batch_size'            : 1024,
    'iters_between_outputs' : 200,

    # Task specs
    'num_rooms'             : 2,
    'num_doors'             : 1,
    'room_widths'           : [6,8],
    'room_heights'          : [6,8],
    'state_size'            : 8,
    'num_actions'           : 5,

}



"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """

    for key, val in updates.items():
        par[key] = val
        print(key, val)

    update_dependencies()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    pass

update_dependencies()

print("--> Parameters successfully loaded.\n")
