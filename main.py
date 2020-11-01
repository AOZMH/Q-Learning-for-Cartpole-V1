import gym
from util import Q_Learner, Cartpole_Discretizer

def train_episode(env, model, params, episode):
    observation = env.reset()
    
    for step_ix in range(1000):
        # e-greedy policy
        action = model.decide_action(observation, train_mode=True)
        # take action
        new_observation, reward, done, info = env.step(action)
        # punish early stop
        if done and step_ix < 180:
            reward = -2
        # update Q-table
        model.update_q_table(observation, action, reward, new_observation)
        # update observation
        observation = new_observation
        
        if done:
            break
    
    return step_ix+1


def eval_episode(env, model, num_trials):
    acc_steps = 0
    succeeded_trials = 0
    
    for trial_ix in range(num_trials):
        observation = env.reset()
        for step_ix in range(1000):
            # e-greedy policy
            action = model.decide_action(observation, train_mode=False)
            # take action
            new_observation, reward, done, info = env.step(action)
            # punish early stop
            if done:
                succeeded_trials += int(step_ix>=199)
                break
            # update observation
            observation = new_observation
        acc_steps += step_ix + 1
    
    print('Evaluationg results:')
    print('Averate steps: {}. Succeeded trials: {}/{}'.format(acc_steps / num_trials, succeeded_trials, num_trials))
    return succeeded_trials


def run(params):
    env = gym.make('CartPole-v0')
    discretizer = Cartpole_Discretizer(params)
    num_states = params['cart_pos_num'] * params['cart_v_num'] * params['pole_angle_num'] * params['pole_v_num']
    model = Q_Learner(num_states, env.action_space.n, discretizer, params)

    # training
    if params['train_or_eval'] == 'train':
        acc_steps = 0
        best_suc_rate = 0.
        for episode in range(params['num_episode']):
            step_ix = train_episode(env, model, params, episode)
            acc_steps += step_ix

            if (episode+1) % params['print_every'] == 0:
                print('Episode {}/{}, stopped with {} steps'.format(episode+1, params['num_episode'], acc_steps/params['print_every']))
                acc_steps = 0

            if (episode+1) % params['eval_every'] == 0:
                succeeded_trials = eval_episode(env, model, params['num_trials'])
                if succeeded_trials / params['num_trials'] > best_suc_rate:
                    best_suc_rate = succeeded_trials / params['num_trials']
                    model.save_q_table(params['checkpoint_q_table'].replace('.npy', ''))
                print('Best success rate: {}.'.format(best_suc_rate))
    
    # evaluating
    elif params['train_or_eval'] == 'eval':
        model.load_q_table(params['checkpoint_q_table'])
        eval_episode(env, model, params['num_trials'])


if __name__ == "__main__":
    params = {
        'train_or_eval': 'train',
        'num_episode': 500000,
        'eval_every': 1000,
        'print_every': 100,
        'num_trials': 1000,
        'lr': 0.2,
        'eps': 0.05,
        'gamma': 0.95,
        'cart_pos_num': 8,
        'cart_v_num': 8,
        'pole_angle_num': 8,
        'pole_v_num': 8,
        'checkpoint_q_table': './data/q_table_1102.npy'
    }
    run(params)
