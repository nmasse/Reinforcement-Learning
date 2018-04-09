"""
Nicolas Masse and Gregory Grant, 2018
"""

import tensorflow as tf
import numpy as np
import AdamOpt
from parameters import *
import stimulus
import matplotlib.pyplot as plt
import os, sys, time, contextlib
from PIL import Image

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
        self.screen_history = []
        self.prediction_history = []
        self.action_history = []
        self.reward_history = []
        self.completion_history = []
        self.error_history = []
        self.hidden_history = []

        # Setting up network shape
        if par['atari']:
            placeholder = tf.zeros([par['batch_train_size'], *par['downsampled_shape']])
            self.n_input = self.convolutional(placeholder).shape[1]
        else:
            self.n_input = np.product(par['observation_shape'])

        # Initializing internal states
        self.default_state = tf.zeros([par['batch_train_size'], self.n_input])
        self.pred_state = tf.zeros([par['batch_train_size'], self.n_input])
        self.rnn_state = par['h_init']
        self.rnn_shape = self.rnn_state.shape

        # Run and optimize
        self.initialize_variables()
        self.run_model()
        self.optimize()

        with tf.device("/cpu:0"):
            self.reset = tf.py_func(self.stim.ensemble_reset, [], [])


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


    def convolutional(self, conv0):

        rinit = tf.random_uniform_initializer
        c = 0.02

        for i in range(3):

            filters = 32 if i < 2 else 16

            conv1 = tf.layers.conv2d(conv0, filters=filters, kernel_size=2, \
                    strides=1, padding='valid', data_format='channels_last',
                    activation=tf.nn.relu, kernel_initializer=rinit(-c,c),
                    bias_initializer=rinit(-c,c), trainable=True)

            conv2 = tf.layers.conv2d(conv1, filters=filters, kernel_size=2, \
                    strides=1, padding='valid', data_format='channels_last',
                    activation=tf.nn.relu, kernel_initializer=rinit(-c,c),
                    bias_initializer=rinit(-c,c), trainable=True)

            conv0 = tf.layers.max_pooling2d(conv2, (3,3), strides=3, \
                    padding='valid', data_format='channels_last')

        return tf.reshape(conv0, [par['batch_train_size'], -1])


    def interact(self, action):
        if par['action_type'] == 'continuum':
            act_eff = action*(par['action_set'][1][0] - par['action_set'][0][0]) - par['action_set'][0][0]
        elif par['action_type'] == 'discrete':
            act_eff = np.int32(np.argmax(action, axis=1))

        act_eff = np.reshape(act_eff, par['batch_train_size'])
        screen, obs, rew, done = self.stim.run_step(act_eff)
        return [np.float32(screen), np.float32(obs), np.float32(rew), np.float32(done)]

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
            stim = target if t != 0 else self.default_state
            target = obs if t != 0 else self.default_state

            # Calculate output
            total_error, rnn_state, action_state = self.network(stim, target)

            # Placeholder operation
            if par['action_type'] == 'continuum':
                action_state = tf.nn.sigmoid(action_state)

            # Step the environment
            with tf.device("/cpu:0"):
                screen, obs, rew, done = tf.py_func(self.interact, [action_state], [tf.float32]*4)

            # Explicitly set observation shape (not required, but recommended)
            obs.set_shape([par['batch_train_size'], *par['downsampled_shape']])
            screen.set_shape([par['trials_to_animate'], *par['observation_shape']])

            # Log network state
            self.error_history.append(total_error)
            self.hidden_history.append(rnn_state)

            # Log environment and action state
            self.screen_history.append(screen)
            self.prediction_history.append(self.pred_state)
            self.action_history.append(action_state)
            self.reward_history.append(rew)
            self.completion_history.append(done)

            # The next two steps have been placed after the logging
            # step for animation viewing after the model is run.

            # Normalize signal strength
            obs = obs/255

            # Use convolutional network while transitioning to the next
            # time step, if desired.
            if par['atari']:
                obs = self.convolutional(obs)


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


def main(gpu_id):

    # Select a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    # Start TensorFlow session
    with tf.Session() as sess:
        print('--> Initializing model...')
        with tf.device("/gpu:0"):
            model = AutoModel()
            init = tf.global_variables_initializer()
        sess.run(init)
        print('--> Model successfully initialized.\n')

        # Training Loop
        for i in range(par['num_iterations']):

            # Train network and pull network information
            _, screen, pred, act, rew, comp, err, hid, perf_loss, spike_loss = sess.run([
                    model.train_op, model.screen_history, model.prediction_history, \
                    model.action_history, model.reward_history, model.completion_history,\
                    model.error_history, model.hidden_history, model.error_loss, model.spike_loss])
            _ = sess.run([model.reset])

            # Display network performance
            if i%par['iters_between_outputs'] == 0 and i != 0:
                feedback = ['Iter. ' + str(i), 'Perf. Loss: ' + str(np.round(perf_loss, 4)),
                            'Spike Loss: ' + str(np.round(spike_loss, 4))]
                print(' | '.join(feedback))

                animation(i, screen)

        print('Simulation complete.\n')


def animation(i, observations):

    #shape = [time steps x trials x *frame]
    observations = np.array(observations, dtype=np.uint8)

    # Select a single trial
    for b in range(par['trials_to_animate']):
        obs = observations[:,b,...]

        # Iterate over each frame
        for t in range(par['num_steps']):

            # Export Numpy array (of frame) to image
            im = Image.fromarray(obs[t,...])
            im.save('./anim/_trial{:04d}_frame{:04d}.png'.format(b, t))

        # Use ffmpeg to collect the images into a short animation
        os.system('ffmpeg -nostats -loglevel 0 -r 25 -i ./anim/_trial{0:04d}_frame%04d.png '.format(b) \
        + '-vcodec libx264 -crf 25 -pix_fmt yuv420p ./anim/iter{1}-trial{0}.mp4'.format(b, i))

        # Delete the frames generated to make way for the next batch
        os.system('rm -f ./anim/_*.png')


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit via KeyboardInterrupt')
