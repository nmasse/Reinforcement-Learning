import stimulus
import numpy as np

env = stimulus.RoomStim()

# self.action_dict = {0:'up', 1:'down', 2:'left', 3:'right', 4:'door'}

while True:
    act = input('--> ')
    if act == 'w':
        action = [1,0,0,0,0]
    elif act == 's':
        action = [0,1,0,0,0]
    elif act == 'a':
        action = [0,0,1,0,0]
    elif act == 'd':
        action = [0,0,0,1,0]
    elif act == 'e':
        action = [0,0,0,0,1]
    elif act == 'q':
        quit()

    state, reward = env.action(np.array(action))
    if reward > 0:
        print('Reward!')
