"""
Nicolas Masse and Gregory Grant, 2018
"""

import tensorflow as tf
import numpy as np
import stimulus
import os, sys
import matplotlib.pyplot as plt
from parameters import *
from itertools import product

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model():

    def __init__(self, state, state_list, target_Q_list):

        self.layer_dims = [par['state_size'], *par['n_hidden'], par['num_actions']]
        self.learning_rate = par['learning_rate']
        self.state = state
        self.noise_std = par['noise_std']

        self.state_list = state_list
        self.target_Q_list = target_Q_list

        self.declare_variables()

        self.output, _                = self.run(self.state)
        self.opt_out, self.spike_hist = self.run(self.state_list)

        self.optimize()


    def declare_variables(self):
        c = 0.02

        for n in range(len(self.layer_dims)-1):
            with tf.variable_scope('feedforward'+str(n)):
                tf.get_variable('W', initializer = np.float32(np.random.uniform(-c,c, [self.layer_dims[n+1],self.layer_dims[n]])))
                tf.get_variable('b', initializer = np.zeros((self.layer_dims[n+1], 1), dtype = np.float32))


    def run(self, input_data):
        spike_hist = []
        x = input_data
        for n in range(len(self.layer_dims)-1):
            with tf.variable_scope('feedforward'+str(n), reuse=True):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                if n <  len(self.layer_dims)-2:
                    x = tf.matmul(W, x) + b
                    x = tf.nn.relu(x + tf.random_normal(x.shape, mean=0.0, stddev=self.noise_std))
                    spike_hist.append(x)
                else:
                    x = tf.matmul(W, x) + b
                    x = x + tf.random_normal(x.shape, mean=0.0, stddev=self.noise_std)

        return x, spike_hist


    def optimize(self):

        # Use all trainable variables
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.perf_loss = tf.reduce_mean(tf.square(self.opt_out - self.target_Q_list))
        self.spike_loss = par['spike_cost']*tf.reduce_mean(self.spike_hist)
        self.train_op = opt.minimize(self.perf_loss + self.spike_loss)


def main(gpu_id = 0):

    # Select a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    stim = stimulus.RoomStim()


    state = tf.placeholder(tf.float32, shape = [par['state_size'], 1], name='state')  # input data
    state_list = tf.placeholder(tf.float32, shape = [par['state_size'], par['batch_size']], name='state_list')
    target_Q_list = tf.placeholder(tf.float32, shape = [par['num_actions'], par['batch_size']], name='target_Q_list')

    # Start TensorFlow session
    with tf.Session() as sess:
        print('--> Initializing model...')
        with tf.device("/gpu:0"):
            model = Model(state, state_list, target_Q_list)
        init = tf.global_variables_initializer()
        sess.run(init)
        print('--> Model successfully initialized.\n')

        # Training Loop
        for j in range(par['num_epochs']):

            state_hist = []
            target_Q_hist = []
            counter_hist = []

            while len(state_hist) < 1*par['batch_size']:

                stim.reset_agent()
                counter = 0
                reward = 0

                while counter < par['max_steps'] and reward == 0:

                    current_state = stim.return_state()
                    Q = sess.run(model.output, feed_dict = {state: np.reshape(current_state, (par['state_size'], 1))})
                    repeat = True
                    while repeat:
                        if np.random.rand() < par['epsilon']:
                            action_index = np.random.choice(np.arange(5))
                        else:
                            action_index = np.argmax(Q)
                        new_state, reward = stim.action(action_index)
                        if not all(new_state == current_state):
                            repeat = False

                    """
                        if not all(new_state == current_state):
                            repeat = False
                    """
                    Q_new = sess.run(model.output, feed_dict = {state: np.reshape(new_state, (par['state_size'], 1))})
                    target_Q = np.squeeze(Q)
                    if reward > 0:
                        # terminal state
                        target_Q[action_index] = reward

                    else:
                        target_Q[action_index] = par['gamma']*np.max(Q_new)

                    state_hist.append(current_state)
                    target_Q_hist.append(target_Q)


                    counter += 1
                counter_hist.append(counter)

            # randomly sample from state_list and target_Q_list
            ind = np.uint16(np.random.permutation(len(state_hist)))
            ind = ind[:par['batch_size']]
            state_hist = state_hist[-1024:]
            target_Q_hist = target_Q_hist[-1024:]
            state_hist = np.stack(state_hist, axis = 1)
            target_Q_hist = np.stack(target_Q_hist, axis = 1)

            _, perf_loss, spike_loss = sess.run([model.train_op, model.perf_loss, model.spike_loss],
                                                feed_dict = {state_list: state_hist, target_Q_list: target_Q_hist})
            print('Epoch {:>3} | <Steps> = {:0.1f}, <Q> = {:0.4f} | Perf. = {:.2E}, Spike = {:.2E}'.format(j, \
                np.mean(np.stack(counter_hist)), np.mean(target_Q_hist), perf_loss, spike_loss))


            if j%par['iters_between_outputs'] == 0 and j != 0:
                print('Outputting Q matrix')
                num_rooms = stim.num_rooms
                room_ids = stim.env_data['ids']
                for room_id in room_ids:

                    # Get needed info
                    room_dims = stim.env_data['dims'][:,room_id]
                    locs = [[a,b] for a, b in product(np.arange(room_dims[0]), np.arange(room_dims[1]))]
                    actions = np.arange(par['num_actions'])

                    # Aggregate Q matrices for each location
                    Q_output = np.zeros([room_dims[0], room_dims[1], par['num_actions']])
                    for l in locs:
                        stim.set_agent(inp_id=room_id, loc=l)
                        Q = sess.run(model.output, feed_dict = {state: np.reshape(stim.return_state(), (par['state_size'], 1))})
                        Q_output[l[0],l[1],:] = np.squeeze(Q)

                    # Plot specific actions
                    fig, ax = plt.subplots(3,2,figsize=(8,10))
                    for a, ind in zip(actions, [[0,0],[0,1],[1,0],[1,1],[2,0]]):
                        im = ax[ind[0],ind[1]].imshow(Q_output[:,:,a], cmap='Purples', clim=[np.min(Q_output), np.max(Q_output)])
                        ax[ind[0], ind[1]].set_title('Action "{}"'.format(stim.action_dict[a]))
                        plt.colorbar(im, ax=ax[ind[0], ind[1]])

                    # Room map
                    output = np.zeros([room_dims[0], room_dims[1], 3])
                    door_locs = np.int8(stim.env_data['doors'][room_id])
                    for j in range(len(door_locs)):
                        output[door_locs[j,0], door_locs[j,1],2] = 0.75

                    if room_id == stim.rew_id:
                        output[stim.rew_loc[0],stim.rew_loc[1],0] = 0.75

                    ax[2,1].set_title('Room Map (Door=Blue, Rew=Red)')
                    ax[2,1].imshow(output)

                    # Show plots
                    plt.suptitle('Room {}, Doors at {}'.format(room_id, str(stim.env_data['doors'][room_id])))
                    plt.show()



if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit via KeyboardInterrupt')
