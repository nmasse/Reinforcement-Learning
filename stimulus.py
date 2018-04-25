import numpy as np
import graph as graph_setup
#from parameters import *


class RoomStim:

    def __init__(self):
        self.num_doors = 2
        self.num_rooms = 4
        self.widths = np.arange(3,6)
        self.heights = np.arange(3,6)
        self.random_rooms = False
        self.action_dict = {0:'up', 1:'down', 2:'left', 3:'right', 4:'door'}

        self.rew_id  = 4
        self.rew_loc = [1,1]

        max_doors = self.widths[0]*self.heights[0] - (self.widths[0]-2)*(self.heights[0]-2) - 4
        if self.num_doors > max_doors:
            raise Exception('Too many doors for the smallest room.')

        self.new_environment(num_rooms=self.num_rooms)
        self.reset_agent()


    def action(self, action_vector):

        action = self.move_agent(action_vector)
        print('-->', action, 'to {} in room {}\n'.format(self.loc, self.id))
        self.get_state()
        self.print_room()

        return self.state, self.reward


    def move_agent(self, action_vector):
        #action_vector = np.exp(action_vector)/np.sum(np.exp(action_vector))
        action = self.action_dict[np.random.choice(np.arange(len(action_vector)), p=action_vector)]

        if action == 'up':
            self.loc[0] = np.max([0, self.loc[0]-1])
        elif action == 'down':
            self.loc[0] = np.min([self.env_data['dims'][0,self.id]-1, self.loc[0]+1])
        elif action == 'left':
            self.loc[1] = np.max([0, self.loc[1]-1])
        elif action == 'right':
            self.loc[1] = np.min([self.env_data['dims'][1,self.id]-1, self.loc[1]+1])
        elif action == 'door':
            if self.loc in self.env_data['doors'][self.id,:].tolist():
                print('-'*40)
                ind = self.env_data['doors'][self.id,:].tolist().index(self.loc)
                self.id = self.env_data['graph'][self.id][ind]
                self.loc = np.int8(self.env_data['doors'][self.id,ind]).tolist()
            else:
                pass    # There is no door here

        return action


    def get_state(self):

        # Check for a door
        if self.loc == self.env_data['doors'][self.id,0].tolist() or self.loc == self.env_data['doors'][self.id,1].tolist():
            door = np.ones(1)
        else:
            door = np.zeros(1)

        # Find walls in x and y axes
        max_locs = self.env_data['dims'][:,self.id] - 1
        w1 = max_locs - np.array(self.loc)
        w2 = np.array(self.loc) - np.zeros(2, dtype=np.int8)

        # Determine room color
        color = np.array(mod8_base2(self.id), dtype=np.float32)

        # Build up state vector
        self.state = np.concatenate([w1, w2, door, color])

        # Check for reward
        if self.id == self.rew_id and self.loc == self.rew_loc:
            self.reward = 1.
        else:
            self.reward = 0.


    def make_rooms(self, num_rooms):
        room_ids, room_graph = self.graph_rooms()
        room_widths = np.random.choice(self.widths, num_rooms)
        room_heights = np.random.choice(self.heights, num_rooms)
        door_locs = np.zeros([num_rooms, self.num_doors, 2])

        for i in room_ids:
            w = room_widths[i]
            h = room_heights[i]

            locs1 = np.stack([np.arange(1,w-1), np.full(w-2,0)]).T.tolist()
            locs2 = np.stack([np.arange(1,w-1), np.full(w-2,h-1)]).T.tolist()
            locs3 = np.stack([np.full(h-2,0), np.arange(1,h-1)]).T.tolist()
            locs4 = np.stack([np.full(h-2,w-1), np.arange(1,h-1)]).T.tolist()
            locs = locs1 + locs2 + locs3 + locs4

            loc_set = np.random.choice(len(locs), self.num_doors, replace=False)
            door_locs[i,:,:] = [locs[loc_set[n]] for n in range(self.num_doors)]

        room_dims = np.stack([room_widths, room_heights])

        return room_ids, room_dims, door_locs, room_graph


    def graph_rooms(self):
        room_ids, graph = graph_setup.make_new_graph(self.num_rooms, self.num_doors)

        return room_ids, graph


    def print_room(self):
        locs = np.int8(self.env_data['doors'][self.id])
        out = np.zeros(self.env_data['dims'][:,self.id])

        # Show doors
        for j in range(self.num_doors):
            out[locs[j,0], locs[j,1]] = 1

        # Show reward
        if self.id == self.rew_id:
            out[self.rew_loc[0], self.rew_loc[1]] = 3

        # Show agent
        out[self.loc[0], self.loc[1]] = 2
        print(np.int8(out))
        print('S:', self.state, '\n')


    def new_environment(self, num_rooms):
        ids, dims, doors, graph = self.make_rooms(num_rooms)

        self.env_data = {
            'ids'   : ids,
            'dims'  : dims,
            'doors' : doors,
            'graph' : graph
        }


    def reset_agent(self):

        self.id = np.random.choice(self.env_data['ids'])
        self.loc = [np.random.choice(self.env_data['dims'][0,self.id]), np.random.choice(self.env_data['dims'][1,self.id])]


def mod8_base2(n):
    n = n%8
    out = []
    for i in range(3):
        out.append(0 if n%2 == 0 else 1)
        n = n // 2

    return out[::-1]



r = RoomStim()
for i in range(10):
    act = np.random.rand(5)
    act = np.exp(act)/np.sum(np.exp(act))
    r.action(act)

print('-'*70)
r.reset_agent()
for i in range(10):
    act = np.random.rand(5)
    act = np.exp(act)/np.sum(np.exp(act))
    r.action(act)
