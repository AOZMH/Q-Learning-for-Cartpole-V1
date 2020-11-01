# Utilizations for Cartpole Q-learning
import random
import numpy as np

class Q_Learner(object):

    def __init__(self, num_states, num_actions, discretizer, params):
        '''
        num_states, num_actions: 状态与动作空间数量
        discretizer: 离散化状态的函数，给定状态返回状态空间内的一个int
        params: 包括lr, epsilon等学习/预测相关参数
        '''
        self.discretizer = discretizer
        self.q_table = np.random.uniform(low=0, high=1, size=(num_states, num_actions))
        self.lr = params['lr']      # learning rate when updating q-table
        self.eps = params['eps']    # e-greedy parameter
        self.gamma = params['gamma']  # discount factor
    
    def decide_action(self, observation, train_mode=False):
        
        state = self.discretizer(observation)
        action_space = range(self.q_table.shape[-1])
        if random.random() < self.eps and train_mode:   # when evaluating, do not do random (?)
            return random.choice(action_space)
        else:
            return self.q_table[state].argmax()
    
    def update_q_table(self, observation_1, action, reward, observation_2):
        
        state_1 = self.discretizer(observation_1)
        state_2 = self.discretizer(observation_2)

        max_q = self.q_table[state_2].max()

        # Update q-table with lr
        self.q_table[state_1][action] = \
            (1-self.lr) * self.q_table[state_1][action] + \
            self.lr * (reward + self.gamma * max_q)
    
    def save_q_table(self, fn_out):
        np.save(fn_out, self.q_table)
    
    def load_q_table(self, fn_in):
        self.q_table = np.load(fn_in)
        

class Cartpole_Discretizer(object):
    
    def __init__(self, range_nums):
        
        # range numbers
        self.cart_pos_num = range_nums['cart_pos_num']
        self.cart_v_num = range_nums['cart_v_num']
        self.pole_angle_num = range_nums['pole_angle_num']
        self.pole_v_num = range_nums['pole_v_num']
        
        # pre-compute digitization ranges
        self.cart_pos_ranges = \
            np.linspace(-2.4, 2.4, range_nums['cart_pos_num']+1)[1:-1]
        self.cart_v_ranges = \
            np.linspace(-3, 3, range_nums['cart_v_num']+1)[1:-1]
        self.pole_angle_ranges = \
            np.linspace(-0.5, 0.5, range_nums['pole_angle_num']+1)[1:-1]
        self.pole_v_ranges = \
            np.linspace(-3, 3, range_nums['pole_v_num']+1)[1:-1]

    def __call__(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        
        # convert to the index in discretized ranges
        cart_pos_ix = np.digitize(cart_pos, self.cart_pos_ranges)
        cart_v_ix = np.digitize(cart_v, self.cart_v_ranges)
        pole_angle_ix = np.digitize(pole_angle, self.pole_angle_ranges)
        pole_v_ix = np.digitize(pole_v, self.pole_v_ranges)
        #print(cart_pos_ix, cart_v_ix, pole_angle_ix, pole_v_ix)

        # convert all ix to unique state index
        cur_state = cart_pos_ix
        cur_state = self.cart_pos_num * cur_state + cart_v_ix
        cur_state = self.cart_v_num * cur_state + pole_angle_ix
        cur_state = self.pole_angle_num * cur_state + pole_v_ix
        #print(cur_state)

        return cur_state
