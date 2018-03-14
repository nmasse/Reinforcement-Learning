"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import AdamOpt
from parameters import *
import stimulus
import matplotlib.pyplot as plt
import os, sys, time

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

        # Initializing internal states
        self.rnn_state = par['h_init'][0]
        self.pred_state = tf.zeros(par['observation_shape'])
        self.rnn_shape = self.rnn_state.shape

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()


    def interact(self, action):
        act_eff = np.int32(np.argmax(action, axis=0))
        obs, rew, done = self.stim.run_step(act_eff)
        return [np.float32(obs), np.float32(rew), np.float32(done)]


    def initialize_variables(self):
        c = 0.02
        for lid in range(1):
            with tf.variable_scope('layer'+str(lid)):
                tf.get_variable('W_err1', shape=[par['n_hidden'][lid],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(0, c), trainable=True)
                tf.get_variable('W_err2', shape=[par['n_hidden'][lid],par['n_dendrites'],par['n_input']],
                                initializer=tf.random_uniform_initializer(0, c), trainable=True)

                tf.get_variable('W_pred', shape=[par['n_input'],par['n_hidden'][lid]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('W_act', shape=[par['n_output'],par['n_hidden'][lid]],
                                initializer=tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('b_pred', initializer = np.zeros((par['n_input'],1), dtype = np.float32), trainable=True)
                tf.get_variable('b_act', initializer = np.zeros((par['n_output'],1), dtype = np.float32), trainable=True)

                tf.get_variable('W_rnn', shape = [par['n_hidden'][lid], par['n_dendrites'], par['n_hidden'][lid]], initializer = tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('b_rnn', initializer = np.zeros((par['n_hidden'][lid],1), dtype = np.float32), trainable=True)


    def calc_error(self, target, prediction):
        return tf.nn.relu(prediction - target), tf.nn.relu(target - prediction)


    def layer(self, target, time, lid=0):
        # Loading all weights and biases
        with tf.variable_scope('layer'+str(lid), reuse=True):
            W_err1 = tf.get_variable('W_err1') #positive error
            W_err2 = tf.get_variable('W_err2') #negative error
            W_pred = tf.get_variable('W_pred')
            b_pred = tf.get_variable('b_pred')
            W_act  = tf.get_variable('W_act')
            b_act  = tf.get_variable('b_act')
            W_rnn  = tf.get_variable('W_rnn')
            b_rnn  = tf.get_variable('b_rnn')


        # Masking certain weights
        #W_rnn *= tf.constant(par['w_rnn_mask'], dtype=tf.float32)
        #if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            #W_rnn = tf.tensordot(tf.nn.relu(W_rnn), tf.constant(par['EI_matrix']), [[2],[0]])

        # Processing data for RNN step
        err_stim1, err_stim2 = self.calc_error(target, self.pred_state)

        inp_act = tf.tensordot(W_err1, err_stim1, [[2],[0]]) + tf.tensordot(W_err2, err_stim2, [[2],[0]]) # Error activity
        rnn_act = tf.tensordot(W_rnn, self.rnn_state, [[2],[0]])       # RNN activity
        tot_act = par['alpha_neuron']*(inp_act + rnn_act)         # Modulating
        act_eff = tf.reduce_sum(tot_act, axis=1) # Summing dendrites

        # Placeholder for later development
        rnn_next = 0.

        # Updating RNN state
        self.rnn_state = tf.nn.relu(self.rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next  + b_rnn \
            + tf.random_normal(self.rnn_shape, 0, par['noise_rnn'], dtype=tf.float32))

        # Prediction state
        self.pred_state = tf.nn.relu(tf.matmul(W_pred, self.rnn_state) + b_pred)

        # Action state
        self.act_state = tf.nn.relu(tf.matmul(W_act, self.rnn_state) + b_act)

        return err_stim1 + err_stim2, self.rnn_state, self.act_state


    def run_model(self):

        # Iterate through time via the input data
        for t in range(par['num_steps']):
            print(t, end='\r')
            stim = obs if t != 0 else tf.zeros(par['observation_shape'])

            total_error, rnn_state, action_state = self.layer(stim, t)
            obs, rew, done = tf.py_func(self.interact, [action_state], [tf.float32,tf.float32,tf.float32])

            self.error_history.append(total_error)
            self.hidden_history.append(rnn_state)

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
        #self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.abs(self.hidden_history))

        # Build train operation
        self.loss = self.error_loss # + self.spike_loss
        grads_and_vars = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(grads_and_vars)


def main():

    """ Reset TensorFlow before running anything """
    tf.reset_default_graph()

    """ Load stimulus """


    """ Define placeholders """
    inp = tf.placeholder(tf.float32, [16,4])


    """ Start TensorFlow session """
    with tf.Session() as sess:
        model = AutoModel()

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)




        out = sess.run([model.train_op, model.observation_history, model.prediction_history, \
                model.action_history, model.reward_history, model.completion_history,\
                model.error_history, model.hidden_history])
        print(out[1])
        print('Success!')


main()
