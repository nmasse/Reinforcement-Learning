"""
Nicolas Masse and Gregory Grant, 2018
"""

import tensorflow as tf
import numpy as np
import stimulus
import os, sys
import matplotlib.pyplot as plt
from itertools import product


state_size = 8
num_actions = 5
batch_size = 1024
num_epochs = 10001
num_iterations = 10
gamma = 0.75
#T = 10.
epsilon = 0.5

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model():

    def __init__(self, state, state_list, target_Q_list):

        self.layer_dims = [state_size, 200, 200, num_actions]
        self.learning_rate = 0.0005
        self.state = state

        self.state_list = state_list
        self.target_Q_list = target_Q_list

        self.run()

        self.optimize()

    def run(self):

        c = 0.02
        x = self.state
        for n in range(len(self.layer_dims)-1):
            with tf.variable_scope('feedforward'+str(n)):
                W = tf.get_variable('W', initializer = np.float32(np.random.uniform(-c,c, [self.layer_dims[n+1],self.layer_dims[n]])))
                b = tf.get_variable('b', initializer = np.zeros((self.layer_dims[n+1], 1), dtype = np.float32))
                if n <  len(self.layer_dims)-2:
                    x = tf.nn.relu(tf.matmul(W, x) + b)
                else:
                    x = tf.matmul(W, x) + b

        self.output = x


    def optimize(self):

        # Use all trainable variables
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)


        x = self.state_list
        for n in range(len(self.layer_dims)-1):
            with tf.variable_scope('feedforward'+str(n), reuse = True):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                if n <  len(self.layer_dims)-2:
                    x = tf.nn.relu(tf.matmul(W, x) + b)
                else:
                    x = tf.matmul(W, x) + b

        print('x',x)
        print('self.target_Q_list',self.target_Q_list)
        loss = tf.reduce_mean(tf.square(x - self.target_Q_list))
        self.train_op = opt.minimize(loss)


def main(gpu_id = 0):

    # Select a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow before running anything
    tf.reset_default_graph()

    stim = stimulus.RoomStim()


    state = tf.placeholder(tf.float32, shape = [state_size, 1], name='state')  # input data
    state_list = tf.placeholder(tf.float32, shape = [state_size, batch_size], name='state_list')
    target_Q_list = tf.placeholder(tf.float32, shape = [num_actions, batch_size], name='target_Q_list')
    epsilon = 0.1

    # Start TensorFlow session
    with tf.Session() as sess:
        print('--> Initializing model...')
        with tf.device("/gpu:0"):
            model = Model(state, state_list, target_Q_list)
        init = tf.global_variables_initializer()
        sess.run(init)
        print('--> Model successfully initialized.\n')

        # Training Loop
        for j in range(num_epochs):

            state_hist = []
            target_Q_hist = []
            counter_hist = []

            if j>50:
                epsilon = 0.1
            elif j>100:
                epsilon = 0.1
            elif j>4000:
                epsilon = 0.0001

            while len(state_hist) < 1*batch_size:

                stim.reset_agent()
                counter = 0
                reward = 0

                while counter < 99 and reward == 0:

                    current_state = stim.return_state()
                    Q = sess.run(model.output, feed_dict = {state: np.reshape(current_state, (state_size, 1))})
                    #Q_softmax = np.exp(T*Q)/np.sum(np.exp(T*Q))
                    repeat = True
                    while repeat:
                        if np.random.rand()< epsilon:
                            action_index = np.random.choice(np.arange(5))
                        else:
                            action_index = np.argmax(Q)
                        new_state, reward = stim.action(action_index)
                        if not all(new_state == current_state):
                            repeat = False
                        #print(action_index, Q)
                    #action_index = np.random.choice(np.arange(num_actions), p = np.squeeze(Q_softmax))

                    """
                        if not all(new_state == current_state):
                            repeat = False
                    """
                    Q_new = sess.run(model.output, feed_dict = {state: np.reshape(new_state, (state_size, 1))})
                    target_Q = np.squeeze(Q)
                    if reward > 0:
                        # terminal state
                        target_Q[action_index] = reward

                    else:
                        target_Q[action_index] = gamma*np.max(Q_new)

                    state_hist.append(current_state)
                    target_Q_hist.append(target_Q)
                    #print(current_state, action_index, new_state)


                    counter += 1
                #print(counter, reward, )
                counter_hist.append(counter)

            # randomly sample from state_list and target_Q_list
            ind = np.uint16(np.random.permutation(len(state_hist)))
            ind = ind[:batch_size]
            #print(type(ind))
            #print(len(state_hist))
            state_hist = state_hist[-1024:]
            target_Q_hist = target_Q_hist[-1024:]
            #print(len(target_Q_hist), target_Q_hist[0].shape)
            state_hist = np.stack(state_hist, axis = 1)
            target_Q_hist = np.stack(target_Q_hist, axis = 1)

            #print(state_hist.shape, target_Q_hist.shape)

            _ = sess.run(model.train_op, feed_dict = {state_list: state_hist, target_Q_list: target_Q_hist})
            print('Epoch ', j, ' - mean counter ', np.mean(np.stack(counter_hist)) , ' mean Q ', np.mean(target_Q_hist))


            if j%50 == 0:
                print('Outputting Q matrix')
                print('Reward Location: Room {} at position {}'.format(stim.rew_id, stim.rew_loc))
                num_rooms = stim.num_rooms
                room_ids = stim.env_data['ids']
                for room_id in room_ids:
                    room_dims = stim.env_data['dims'][:,room_id]
                    locs = [[a,b] for a, b in product(np.arange(room_dims[0]), np.arange(room_dims[1]))]
                    actions = np.arange(5)

                    fig, ax = plt.subplots(3,2,figsize=(8,10))
                    for a, ind in zip(actions, [[0,0],[0,1],[1,0],[1,1],[2,0]]):

                        Q_output = np.zeros([room_dims[0], room_dims[1]])
                        for l in locs:
                            stim.set_agent(inp_id=room_id, loc=l)
                            current_state = stim.return_state()
                            Q = sess.run(model.output, feed_dict = {state: np.reshape(current_state, (state_size, 1))})
                            Q_output[l[0],l[1]] = Q[a]

                        im = ax[ind[0], ind[1]].imshow(Q_output, cmap='Purples', clim=[np.min(Q_output), np.max(Q_output)])
                        ax[ind[0], ind[1]].set_title('Action "{}"'.format(stim.action_dict[a]))
                        plt.colorbar(im, ax=ax[ind[0], ind[1]])

                    output = np.zeros([room_dims[0], room_dims[1], 3])
                    door_locs = np.int8(stim.env_data['doors'][room_id])
                    for j in range(len(door_locs)):
                        output[door_locs[j,0], door_locs[j,1],2] = 0.75

                    if room_id == stim.rew_id:
                        output[stim.rew_loc[0],stim.rew_loc[1],0] = 0.75

                    ax[2,1].set_title('Room Map (Door=Blue, Rew=Red)')
                    ax[2,1].imshow(output)
                    plt.suptitle('Room {}, Doors at {}'.format(room_id, str(stim.env_data['doors'][room_id])))
                    plt.show()




if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit via KeyboardInterrupt')
