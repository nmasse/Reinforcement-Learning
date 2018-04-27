import stimulus
import numpy as np

env = stimulus.RoomStim()

# self.action_dict = {0:'up', 1:'down', 2:'left', 3:'right', 4:'door'}

while True:
    act = input('--> ')
    if act == 'w':
        action = 0
    elif act == 's':
        action = 1
    elif act == 'a':
        action = 2
    elif act == 'd':
        action = 3
    elif act == 'e':
        action = 4
    elif act == 'q':
        quit()

    state, reward = env.action(action)
    if reward > 0:
        print('Reward!')
