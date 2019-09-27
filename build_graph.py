import numpy as np
import tensorflow as tf

# build the action operation, state value operation, and the training loss optimization operation
def build_train(model, 
                num_actions,
                lr,
                epsilon,
                nenvs,
                step_size, # batchsize
                lstm_unit=256,
                state_shape=[84, 84, 1],
                grad_clip=40.0,
                value_factor=0.5,
                entropy_factor=0.01,
                continuous=False,
                scope='ppo',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse): # all in the variable scope
        # placeholers
        step_obs_input = tf.placeholder(
            tf.float32, [nenvs] + state_shape, name='step_obs') # agent observe states of [all n envs]
        train_obs_input = tf.placeholder(
            tf.float32, [nenvs*step_size] + state_shape, name='train_obs')
        rnn_state_ph = tf.placeholder(
            tf.float32, [nenvs, lstm_unit*2], name='rnn_state')
        returns_ph = tf.placeholder(tf.float32, [None], name='returns')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        masks_ph = tf.placeholder(tf.float32, [nenvs * step_size], name='masks')
        if continuous:
            actions_ph = tf.placeholder(
                tf.float32, [nenvs*step_size, num_actions], name='action')
            old_log_probs_ph = tf.placeholder(
                tf.float32, [nenvs*step_size, num_actions], name='old_log_prob')
        else:
            actions_ph = tf.placeholder(
                tf.int32, [nenvs*step_size], name='action')
            old_log_probs_ph = tf.placeholder(
                tf.float32, [nenvs*step_size], name='old_log_prob')

        # network outputs for inference
        # all are tf operations
        step_dist, step_value, state_out = model( # mlp network for continuous action space, cnn network for discrete
            step_obs_input, tf.constant(0.0, shape=[nenvs, 1]), rnn_state_ph, # tf.constant(0.0, shape=[nenvs, 1]): lstm mask, rnn_state_ph: rnn state, 
            num_actions, lstm_unit, nenvs, 1, scope='model') # 1: stepsize (used in lstm)
        # e.g. step_dist: if nenv=10
            # [[0.25060725 0.2486053  0.24805398 0.2527335 ]
            # [0.25035715 0.24834014 0.24851206 0.25279066]
            # [0.250718   0.2484012  0.24813466 0.25274616]
            # [0.25038743 0.24846208 0.24854583 0.2526047 ]
            # [0.25027004 0.24865392 0.24861902 0.25245705]
            # [0.25031397 0.2488611  0.24820198 0.25262296]
            # [0.2503548  0.2486403  0.24850756 0.25249735]
            # [0.25038743 0.24846208 0.24854583 0.2526047 ]
            # [0.25029618 0.24864575 0.24842171 0.25263634]
            # [0.25037965 0.24874271 0.24840613 0.25247154]]

        
        # network outputs for training
        train_dist, train_value, _ = model(
            train_obs_input, tf.reshape(masks_ph, [nenvs * step_size, 1]),
            rnn_state_ph, num_actions, lstm_unit, nenvs, step_size,
            scope='model')

        # network weights
        network_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # loss
        advantages = tf.reshape(advantages_ph, [-1, 1])
        returns = tf.reshape(returns_ph, [-1, 1])
        with tf.variable_scope('value_loss'):
            value_loss = tf.reduce_mean(tf.square(returns - train_value))
            value_loss *= value_factor
        with tf.variable_scope('entropy'):
            entropy = tf.reduce_mean(train_dist.entropy())
            entropy *= entropy_factor
        with tf.variable_scope('policy_loss'):
            log_prob = train_dist.log_prob(actions_ph) # log probability of choosing action action_ph. log pi(s, a)/log pi(a|s)
            ratio = tf.exp(log_prob - old_log_probs_ph) # exp(log(x)) = x
            if continuous:
                ratio = tf.reduce_mean(ratio, axis=1, keep_dims=True)
            else:
                ratio = tf.reshape(ratio, [-1, 1])
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(
                ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
            surr = tf.minimum(surr1, surr2) # ppo loss function
            policy_loss = tf.reduce_mean(surr)
        loss = value_loss - policy_loss - entropy

        # gradients
        gradients = tf.gradients(loss, network_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip) # gradients may shrunk

        # update
        grads_and_vars = zip(clipped_gradients, network_vars)
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)
        optimize_expr = optimizer.apply_gradients(grads_and_vars) # compute the gradients of loss, and update params using gradients

        # action
        action = step_dist.sample(1)[0] # 1 is the shape (sample one action), [0] is because the return is a tensor, with shape (1,)
        log_policy = step_dist.log_prob(action) # log probaility of choosing above action

        def train(obs, actions, returns, advantages, log_probs,  rnn_state, masks):
            feed_dict = {
                train_obs_input: obs,
                actions_ph: actions,
                returns_ph: returns,
                advantages_ph: advantages,
                old_log_probs_ph: log_probs,
                rnn_state_ph: rnn_state,
                masks_ph: masks
            }
            sess = tf.get_default_session() #!!! first, get the session
            return sess.run([loss, optimize_expr], feed_dict=feed_dict)[0]

        def act(obs, rnn_state):
            feed_dict = {
                step_obs_input: obs,
                rnn_state_ph: rnn_state,
            }
            sess = tf.get_default_session() #!!! get the session
            ops = [action, log_policy, step_value, state_out] # step_value: V(s)
            return sess.run(ops, feed_dict=feed_dict)

    return act, train
