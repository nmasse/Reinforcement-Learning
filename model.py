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

try:
    import gym
except ModuleNotFoundError:
    quit('Must use python3.5 due to gym not being installed in Anaconda.')

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class AutoModel:

    def __init__(self):

        self.stim = stimulus.GymStim()
        obs, rew, done = self.stim.run_step(tf.random_)

        #self.observation_history = tf.zeros([par['num_steps'], obs.shape])
        #self.prediction_history = tf.zeros([par['num_steps'], obs.shape])
        #self.action_history = tf.zeros([par['num_steps'], obs.shape])

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()


    def initialize_variables(self):
        c = 0.02
        for lid in range(self.num_layers):
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
                tf.get_variable('b_act', initializer = np.zeros((par['n_input'],1), dtype = np.float32), trainable=True)

                tf.get_variable('W_rnn', shape = [par['n_hidden'][lid], par['n_dendrites'], par['n_hidden'][lid]], initializer = tf.random_uniform_initializer(-c, c), trainable=True)
                tf.get_variable('b_rnn', initializer = np.zeros((par['n_hidden'][lid],1), dtype = np.float32), trainable=True)

    def calc_error(self, target, prediction):
        return tf.nn.relu(prediction - target), tf.nn.relu(target - prediction)


    def layer(self, lid, target, time):

        # Loading all weights and biases
        with tf.variable_scope('layer'+str(lid), reuse=True):
            W_err1 = tf.get_variable('W_err1') #positive error
            W_err2 = tf.get_variable('W_err2') #negative error
            W_latent_mu = tf.get_variable('W_latent_mu')
            W_latent_sigma = tf.get_variable('W_latent_sigma')
            W_pred1 = tf.get_variable('W_pred1')
            b_pred1 = tf.get_variable('b_pred1') # prediction bias
            #W_pred2 = tf.get_variable('W_pred2')
            #b_pred2 = tf.get_variable('b_pred2') # prediction bias
            W_rnn  = tf.get_variable('W_rnn')
            b_rnn  = tf.get_variable('b_rnn')


            if lid < self.num_layers - 1 and False:
                W_top_down = tf.get_variable('W_top_down')

        # Masking certain weights
        #W_rnn *= tf.constant(par['w_rnn_mask'], dtype=tf.float32)
        #if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            #W_rnn = tf.tensordot(tf.nn.relu(W_rnn), tf.constant(par['EI_matrix']), [[2],[0]])

        # Processing data for RNN step
        # Currently, input from R l+1 to R l in not weighted
        err_stim1, err_stim2 = self.calc_error(target, self.pred_states[lid])
        rnn_state = self.rnn_states[lid]
        #rnn_state2 = self.rnn_states2[lid]
        if lid != self.num_layers-1 and False:
            rnn_next = tf.matmul(W_top_down, self.rnn_states[lid+1])
        else:
            rnn_next = tf.zeros_like(rnn_state)

        inp_act = tf.tensordot(W_err1, err_stim1, [[2],[0]]) + tf.tensordot(W_err2, err_stim2, [[2],[0]]) # Error activity
        rnn_act = tf.tensordot(W_rnn, rnn_state, [[2],[0]])       # RNN activity
        tot_act = par['alpha_neuron']*(inp_act + rnn_act)         # Modulating
        act_eff = tf.reduce_sum(tot_act, axis=1) # Summing dendrites

        # Updating RNN state
        #rnn_state = self.neuron_td*tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next + b_rnn \
        #    + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))
        rnn_state = tf.nn.relu(rnn_state*(1-par['alpha_neuron']) + act_eff + rnn_next  + b_rnn \
            + tf.random_normal(rnn_state.shape, 0, par['noise_rnn'], dtype=tf.float32))
        self.rnn_states[lid]  = rnn_state

        # Updating prediction state
        latent_mu = tf.matmul(W_latent_mu, rnn_state) # A_hat
        latent_sigma = tf.matmul(W_latent_sigma, rnn_state)
        latent_sample = latent_mu + tf.exp(latent_sigma)*tf.random_normal([par['n_hidden'][lid], par['batch_train_size']], \
            0, 1 , dtype=tf.float32)

        pred1 = tf.nn.relu(tf.matmul(W_pred1, latent_sample)  + b_pred1)
        self.pred_states[lid] = pred1
        #self.pred_states[lid] = tf.nn.relu(tf.matmul(W_pred2, pred1) + b_pred2)

        if lid > - 1:
            self.latent_loss -= 0.5*par['latent_cost']*tf.reduce_sum(1. + latent_sigma - tf.square(latent_mu) - tf.exp(latent_sigma))

        return err_stim1 + err_stim2, rnn_state


    def run_model(self):

        # Start recording the error and hidden state histories
        self.error_history = [[] for _ in range(self.num_layers)]
        self.hidden_history = [[] for _ in range(self.num_layers)]
        self.prediction_history = [[] for _ in range(self.num_layers)]
        self.latent_loss = tf.constant(0.)

        # Iterate through time via the input data
        for t, input_data in enumerate(self.input_data):

            # Iterate over each layer at each time point
            for lid in range(self.num_layers):

                # If the first layer, use the actual input
                # Instead of using desired output for layer, we'll use neuronal input
                stim = input_data if lid == 0 else self.error_history[lid-1][-1]

                # Run the current layer and recover the error matrix,
                # then save the error matrix to the NEXT state position,
                # and record the RNN activity
                layer_error, rnn_state = self.layer(lid, stim, t)

                self.hidden_history[lid].append(rnn_state)
                self.prediction_history[lid].append(self.pred_states[lid])
                self.error_history[lid].append(layer_error)



    def optimize(self):

        # Use all trainable variables
        variables = [var for var in tf.trainable_variables()]
        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])


        self.error_loss = tf.constant(0.)
        for j,eh in enumerate(self.error_history):
            for l in eh:
                self.error_loss += tf.reduce_mean(tf.square(l))

        self.spike_loss = tf.constant(0.)
        for h in self.hidden_history:
            for l in h:
                self.spike_loss += par['spike_cost']*tf.reduce_mean(l)

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.error_loss, self.spike_loss]):
            self.loss = self.error_loss + self.aux_loss + self.spike_loss + self.latent_loss
            self.train_op = adam_optimizer.compute_gradients(self.loss)


def train_and_analyze(gpu_id, save_fn):

    tf.reset_default_graph()
    main(gpu_id, save_fn)
    #update_parameters(revert_analysis_par)


def main(save_fn, gpu_id = None):

    print('\nRunning model.\n')

    ##################
    ### Setting Up ###
    ##################

    """ Set up GPU """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """ Reset TensorFlow before running anything """
    tf.reset_default_graph()

    """ Set up performance recording """
    model_performance = {'accuracy': [], 'par': [], 'task_list': []}

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = AutoModel()
        else:
            with tf.device("/gpu:0"):
                model = AutoModel()

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        #################
        ### Main Loop ###
        #################

        for i in range(par['num_iterations']):

            _, action, perf_loss, spike_loss, hid_hist, pred_hist, err_hist = sess.run([ \
                model.train_op, model.action, model.error_loss, model.spike_loss, \
                model.hidden_history, model.prediction_history, model.error_history])

            if i//par['iters_between_outputs'] == i/par['iters_between_outputs']:
                #print('Iter ', i, 'Perf Loss ', perf_loss, ' AuxLoss ', aux_loss, ' Mean sr ', np.mean(h), ' WL ', wiring_loss)
                print('Iter ', str(i).ljust(3), 'Perf Loss ', perf_loss, ' Mean sr ', np.mean(h[0]), ' WL ', wiring_loss, ' LL ', latent_loss)


    print('\nModel execution complete.')

if __name__ == '__main__':
    main('testing', str(sys.argv[1]))
