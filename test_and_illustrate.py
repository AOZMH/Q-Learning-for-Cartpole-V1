import warnings
warnings.filterwarnings('error')

import gym
from main import run, train_episode, eval_episode
from util import Q_Learner, Cartpole_Discretizer

def print_one_episode(params):
    
    # initialize environment
    env = gym.make('CartPole-v0')
    discretizer = Cartpole_Discretizer(params)
    num_states = params['cart_pos_num'] * params['cart_v_num'] * params['pole_angle_num'] * params['pole_v_num']
    model = Q_Learner(num_states, env.action_space.n, discretizer, params)
    
    # load pretrained model (q-table)
    model.load_q_table(params['checkpoint_q_table'])

    # start one episode and print process to screen
    observation = env.reset()
    for step_ix in range(500000):
        if step_ix % 1000 == 999:
            print('Stepping to {}...'.format(step_ix+1))
        try:
            env.render()    # print process to screen
            action = model.decide_action(observation, train_mode=False)     # e-greedy policy
            new_observation, reward, done, info = env.step(action)  # take action
            if params['end_at_200'] and done:
                break
        except Warning:
            print("Warning catched!")
            step_ix -= 1
            break
        # update observation
        observation = new_observation
    env.close()
    print('Stopped at step {}.'.format(step_ix+1))


if __name__ == "__main__":
    params = {
        'lr': 0.2,
        'eps': 0.05,
        'gamma': 0.95,
        'cart_pos_num': 8,
        'cart_v_num': 8,
        'pole_angle_num': 8,
        'pole_v_num': 8,
        'checkpoint_q_table': './data/q_table_02lr.npy',
        'end_at_200': False,
    }
    print_one_episode(params)
