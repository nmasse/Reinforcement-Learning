"""
Nicolas Masse and Gregory Grant, 2018
"""

import tensorflow as tf
import numpy as np
import AdamOpt
from parameters import *
import stimulus
import matplotlib.pyplot as plt
import os, sys, time

try:
    import gym
except ModuleNotFoundError:
    quit('Must use python3.5 due to gym not being installed in Anaconda.')

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class AutoModel:

    def __init__(self):

        # Opening the gym environment
        self.stim = stimulus.GymStim()

        # Setting up records
        self.observation_history = []
        self.prediction_history = []
        self.action_history = []
        self.reward_history = []
        self.completion_history = []
        self.error_history = []
        self.hidden_history = []

        # Setting up network shape
        self.n_input = np.product(par['observation_shape'])

        # Initializing internal states
        self.rnn_state = par['h_init']
        self.pred_state = tf.zeros([par['batch_train_size'], self.n_input])
        self.rnn_shape = self.rnn_state.shape

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()


    def initialize_variables(self):
        rinit = tf.random_uniform_initializer
        c = 0.02
        with tf.variable_scope('network'):

            # RNN Inputs
            tf.get_variable('W_in', shape = [self.n_input, par['n_dendrites'], par['n_hidden']],
                            initializer = rinit(-c, c), trainable=True)
            tf.get_variable('W_rnn', shape = [par['n_hidden'], par['n_dendrites'], par['n_hidden']],
                            initializer = rinit(-c, c), trainable=True)
            tf.get_variable('W_err1', shape=[self.n_input,par['n_dendrites'],par['n_hidden']],
                            initializer = rinit(0, c), trainable=True)
            tf.get_variable('W_err2', shape=[self.n_input,par['n_dendrites'],par['n_hidden']],
                            initializer = rinit(0, c), trainable=True)
            tf.get_variable('b_rnn', initializer = np.zeros((1,par['n_hidden']), dtype = np.float32), trainable=True)

            # Action Inputs
            tf.get_variable('W_act', shape=[par['n_hidden'],par['n_output']],
                            initializer = rinit(-c, c), trainable=True)
            tf.get_variable('b_act', initializer = np.zeros((1,par['n_output']), dtype = np.float32), trainable=True)

            # Prediction Inputs
            tf.get_variable('W_pred', shape = [par['n_hidden'],self.n_input],
                            initializer = rinit(-c, c), trainable=True)
            tf.get_variable('W_ap', shape = [par['n_output'],self.n_input],
                            initializer =rinit(-c,c), trainable=True)
            tf.get_variable('b_pred', initializer = np.zeros((1,self.n_input), dtype = np.float32), trainable=True)


    def interact(self, action):
        if par['action_type'] == 'continuum':
            act_eff = action*(par['action_set'][1][0] - par['action_set'][0][0]) - par['action_set'][0][0]
        elif par['action_type'] == 'discrete':
            act_eff = np.int32(np.argmax(action, axis=1))

        act_eff = np.reshape(act_eff, par['batch_train_size'])
        obs, rew, done = self.stim.run_step(act_eff)
        return [np.float32(obs), np.float32(rew), np.float32(done)]


    def calc_error(self, target, prediction):
        return tf.nn.relu(prediction - target), tf.nn.relu(target - prediction)


    def network(self, stim, target):

        # Loading all weights and biases
        with tf.variable_scope('network', reuse=True):

            # RNN Inputs
            W_in   = tf.get_variable('W_in')
            W_rnn  = tf.get_variable('W_rnn')
            W_err1 = tf.get_variable('W_err1')
            W_err2 = tf.get_variable('W_err2')
            b_rnn  = tf.get_variable('b_rnn')

            # Action Inputs
            W_act  = tf.get_variable('W_act')
            b_act  = tf.get_variable('b_act')

            # Prediction Inputs
            W_pred = tf.get_variable('W_pred')
            W_ap   = tf.get_variable('W_ap')
            b_pred = tf.get_variable('b_pred')

        # Calculating error from previous time step
        err_stim1, err_stim2 = self.calc_error(target, self.pred_state)

        # Processing data for RNN step
        inp_act = tf.tensordot(stim, W_in, [[1],[0]])                  # Stimulus activity
        err_act = tf.tensordot(err_stim1, W_err1, [[1],[0]]) \
                + tf.tensordot(err_stim2, W_err2, [[1],[0]])           # Error activity
        rnn_act = tf.tensordot(self.rnn_state, W_rnn, [[1],[0]])       # RNN activity
        tot_act = par['alpha_neuron']*(inp_act + err_act + rnn_act)    # RNN modulation
        act_eff = tf.reduce_sum(tot_act, axis=1)                       # Summing dendrites

        # Updating RNN state
        self.rnn_state = tf.nn.relu(self.rnn_state*(1-par['alpha_neuron']) + act_eff  + b_rnn \
            + tf.random_normal(self.rnn_shape, 0, par['noise_rnn'], dtype=tf.float32))

        # Action state
        self.act_state = tf.nn.relu(tf.matmul(self.rnn_state, W_act) + b_act)

        # Prediction state
        self.pred_state = tf.nn.relu(tf.matmul(self.rnn_state, W_pred) \
                        + tf.matmul(self.act_state, W_ap) + b_pred)

        return err_stim1 + err_stim2, self.rnn_state, self.act_state


    def run_model(self):

        # Iterate through time via the input data
        for t in range(par['num_steps']):

            # Call for a stimulus
            stim = target if t != 0 else tf.zeros([par['batch_train_size'], *par['observation_shape']])
            target = obs if t != 0 else tf.zeros([par['batch_train_size'], *par['observation_shape']])

            stim = tf.reshape(stim, [par['batch_train_size'], -1])
            target = tf.reshape(target, [par['batch_train_size'], -1])

            # Calculate output
            total_error, rnn_state, action_state = self.network(stim, target)

            # Placeholder operation
            if par['action_type'] == 'continuum':
                action_state = tf.nn.sigmoid(action_state)

            # Step the environment
            obs, rew, done = tf.py_func(self.interact, [action_state], [tf.float32,tf.float32,tf.float32])

            # Explicitly set observation shape (not required, but recommended)
            obs.set_shape([par['batch_train_size'], *par['observation_shape']])

            # Log network state
            self.error_history.append(total_error)
            self.hidden_history.append(rnn_state)

            # Log environment and action state
            self.observation_history.append(obs)
            self.prediction_history.append(self.pred_state)
            self.action_history.append(action_state)
            self.reward_history.append(rew)
            self.completion_history.append(done)


    def optimize(self):

        # Use all trainable variables
        opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])

        # Calculate losses
        self.error_loss = tf.reduce_mean(tf.square(self.error_history))
        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.abs(self.hidden_history))

        # Build train operation
        self.loss = self.error_loss # + self.spike_loss
        grads_and_vars = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(grads_and_vars)


def main():

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    # Start TensorFlow session
    with tf.Session() as sess:
        print('--> Initializing model...')
        model = AutoModel()

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        print('--> Model successfully initialized.\n')

        # Training Loop
        for i in range(par['num_iterations']):

            # Train network and pull network information
            _, obs, pred, act, rew, comp, err, hid, perf_loss, spike_loss = sess.run([
                    model.train_op, model.observation_history, model.prediction_history, \
                    model.action_history, model.reward_history, model.completion_history,\
                    model.error_history, model.hidden_history, model.error_loss, model.spike_loss])

            # Display network performance
            if i%par['iters_between_outputs'] == 0 and i != 0:
                feedback = ['Iter. ' + str(i), 'Perf. Loss: ' + str(np.round(perf_loss, 4)),
                            'Spike Loss: ' + str(np.round(spike_loss, 4))]
                print(' | '.join(feedback))

        print('Simulation complete.\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        quit('Quit via KeyboardInterrupt')
