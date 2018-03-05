import numpy as np
import gym
from parameters import *

par['num_envs'] = 1024
par['environment'] = 'CartPole-v0'

class GymStim:

    def __init__(self):
        self.ensemble, self.obs, self.rew, self.done, self.act_set = self.create_ensemble()
        self.ensemble_reset()

    def create_ensemble(self):
        sample_env = gym.make(par['environment'])
        sample_env.reset()
        obs, rew, done, info = sample_env.step(sample_env.action_space.sample())
        envtype   = np.dtype(type(sample_env))
        num_acts  = sample_env.action_space.n
        obs_shape = np.shape(obs)

        ensemble  = np.empty(par['num_envs'], dtype=envtype)
        obs       = np.zeros([par['num_envs'], *obs_shape])
        done      = np.full([par['num_envs']], False)
        rew       = np.zeros([par['num_envs']])
        act_set   = np.arange(num_acts)

        for i in range(par['num_envs']):
            ensemble[i] = gym.make(par['environment'])

        return ensemble, obs, rew, done, act_set


    def ensemble_reset(self):
        for i in range(par['num_envs']):
            self.obs[i] = self.ensemble[i].reset()


    def run_step(self, acts=None):
        if acts is None:
            print('Randomizing actions.')
            acts = np.random.choice(self.act_set, par['num_envs'])

        for i in range(par['num_envs']):
            if not self.done[i]:
                self.obs[i], self.rew[i], self.done[i], _ = self.ensemble[i].step(acts[i])
            else:
                self.rew[i] = 0.

        return self.obs, self.rew, self.done

### Minimal Working Example ###

# Create stimulus
stim = GymStim()

# Run a random first step and retrieve the available actions
obs, rew, done = stim.run_step()
actions = stim.act_set

# Now entering the main loop
for t in range(10):
    # The network generates a response (however it decides to)
    network_resp = np.random.choice(actions, par['num_envs'])

    # The response is fed into the simulation, and we get back new data
    obs, rew, done = stim.run_step(network_resp)
    print('\n' + '-'*10+'{}'.format(t)+'-'*10)
    print(obs)  # New observation to give to the network
    print(rew)  # Reward value
