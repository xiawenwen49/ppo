import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import atari_constants
import box_constants
import numpy as np
import tensorflow as tf

from rlsaber.log import TfBoardLogger, dump_constants
from rlsaber.trainer import BatchTrainer
from rlsaber.env import EnvWrapper, BatchEnvWrapper, NoopResetEnv, EpisodicLifeEnv, MaxAndSkipEnv
from rlsaber.preprocess import atari_preprocess
from network import make_network
from agent import Agent
from scheduler import LinearScheduler, ConstantScheduler
from datetime import datetime


def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--load', type=str) # how to load
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true') # training or not training
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env_name = args.env
    tmp_env = gym.make(env_name)
    is_atari = len(tmp_env.observation_space.shape) != 1
    if not is_atari:
        observation_space = tmp_env.observation_space
        constants = box_constants
        if isinstance(tmp_env.action_space, gym.spaces.Box):
            num_actions = tmp_env.action_space.shape[0] # for continuous action space, num_actions means how many continuous actions
        else:
            num_actions = tmp_env.action_space.n # for discrete action space, num_actions means how many selectable actions.
        state_shape = [observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda s: s
        reward_preprocess = lambda r: r / 10.0
        # (window_size, dim) -> (dim, window_size)
        phi = lambda s: np.transpose(s, [1, 0])
    else:
        constants = atari_constants
        num_actions = tmp_env.action_space.n
        state_shape = constants.STATE_SHAPE + [constants.STATE_WINDOW]
        def state_preprocess(state):
            state = atari_preprocess(state, constants.STATE_SHAPE)
            state = np.array(state, dtype=np.float32)
            return state / 255.0
        reward_preprocess = lambda r: np.clip(r, -1.0, 1.0)
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda s: np.transpose(s, [1, 2, 0]) # a transformation function

    # flag of continuous action space
    continuous = isinstance(tmp_env.action_space, gym.spaces.Box) # 'gym.spaces.Box' means continuous action space
    upper_bound = tmp_env.action_space.high if continuous else None

    # save settings
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    model = make_network( # !!! just a lambda function
        constants.CONVS, constants.FCS, use_lstm=constants.LSTM,
        padding=constants.PADDING, continuous=continuous) # model is a function instance, 
                                                        # mlp network for continuous action space, cnn network for discrete

    # learning rate with decay operation
    if constants.LR_DECAY == 'linear':
        lr = LinearScheduler(constants.LR, constants.FINAL_STEP, 'lr')
        epsilon = LinearScheduler(
            constants.EPSILON, constants.FINAL_STEP, 'epsilon')
    else:
        lr = ConstantScheduler(constants.LR, 'lr')
        epsilon = ConstantScheduler(constants.EPSILON, 'epsilon')

    agent = Agent(
        model, # !!!
        num_actions,
        nenvs=constants.ACTORS,
        lr=lr,
        epsilon=epsilon,
        gamma=constants.GAMMA,
        lam=constants.LAM,
        lstm_unit=constants.LSTM_UNIT,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        time_horizon=constants.TIME_HORIZON,
        batch_size=constants.BATCH_SIZE,
        grad_clip=constants.GRAD_CLIP,
        state_shape=state_shape,
        epoch=constants.EPOCH,
        phi=phi,
        use_lstm=constants.LSTM,
        continuous=continuous,
        upper_bound=upper_bound
    )

    saver = tf.train.Saver(max_to_keep=5)
    if args.load:
        saver.restore(sess, args.load)
    else: # this else is important
        sess.run(tf.global_variables_initializer()) # 
    # create environemtns
    envs = []
    for i in range(constants.ACTORS): # 8 actors
        env = gym.make(args.env)
        env.seed(constants.RANDOM_SEED)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env)
            env = EpisodicLifeEnv(env)
        wrapped_env = EnvWrapper(
            env,
            r_preprocess=reward_preprocess,
            s_preprocess=state_preprocess
        ) 
        envs.append(wrapped_env) # append all wrapped_envs
    batch_env = BatchEnvWrapper(envs) # envs is a list

    # sess.run(tf.global_variables_initializer()) # should not be here? otherwise it will override the loaded checkpoint

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.float32)
    end_episode = lambda r, s, e: logger.plot('reward', r, s) # record the reward a episode

    def after_action(state, reward, global_step, local_step):# after an action, check weather need to save model
        # demo mode will not save the model params
        if (global_step % 10**5 >=0 and global_step % 10**5 <= 10 ) and not args.demo : # save model about every 10 ** 5, can't use global step% 10**5 ==0, because global_step may not 
                                                                    # get the number of multiple of 10**5.
            path = os.path.join(outdir, 'model.ckpt')
            print('model saved, global step:{}'.format(global_step))
            saver.save(sess, path, global_step=global_step)

    trainer = BatchTrainer(
        env=batch_env,
        agent=agent, # Agent instannce
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP, # final_step is a total time step limit
        # final_step=12345,
        after_action=after_action, # callback function after an action
        end_episode=end_episode,
        training=not args.demo # if --demo, then not training, if no --demo, then training the policy net and value net
    )
    trainer.start()

if __name__ == '__main__':
    main()
