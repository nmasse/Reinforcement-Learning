import numpy as np
from parameters import *
import contextlib
import gym


class GymStim:

    def __init__(self):
        # Set up a new environment
        self.ensemble, self.obs, self.rew, self.done, self.act_set = self.create_ensemble()
        self.ensemble_reset()

    def create_ensemble(self):

        ensemble  = np.empty(par['batch_train_size'], dtype=par['envtype'])
        obs       = np.zeros([par['batch_train_size'], *par['observation_shape']])
        done      = np.full([par['batch_train_size']], False)
        rew       = np.zeros([par['batch_train_size']])
        act_set   = par['action_set']

        with contextlib.redirect_stdout(None):      # Suppress datatype warnings
            for i in range(par['batch_train_size']):
                ensemble[i] = gym.make(par['environment_type'])

        return ensemble, obs, rew, done, act_set


    def ensemble_reset(self):
        for i in range(par['batch_train_size']):
            self.obs[i] = self.ensemble[i].reset()


    def run_step(self, acts):
        ### THIS IS CALLED FROM WITHIN THE TENSORFLOW GRAPH ###

        for i, act in zip(range(par['batch_train_size']), acts):
            if not self.done[i]:

                # This odd-looking conditional is necessary to
                # prevent weird indexing bugs in the Gym
                if par['action_type'] == 'continuum':
                    self.obs[i], self.rew[i], self.done[i], _ = self.ensemble[i].step(np.array([act]))
                elif par['action_type'] == 'discrete':
                    self.obs[i], self.rew[i], self.done[i], _ = self.ensemble[i].step(act)
            else:
                self.rew[i] = 0.

        return self.obs, self.rew, self.done

"""
###############################
### Minimal Working Example ###
###############################

# Create stimulus
stim = GymStim()

# Run a random first step and retrieve the available actions
obs, rew, done = stim.run_step()
actions = stim.act_set

# Now entering the main loop
for t in range(par['num_steps']):

    # The network generates a response (however it decides to)
    network_resp = np.random.choice(actions, par['batch_train_size'])

    # The response is fed into the simulation, and we get back new data
    obs, rew, done = stim.run_step(network_resp)
    print('\n' + '-'*10+'{}'.format(t)+'-'*10)
    print(obs)  # New observation to give to the network
    print(rew)  # Reward value
"""
