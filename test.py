import gym
# BreakoutNoFrameskip-v4 PongNoFrameskip-v4 CartPole-v0 MountainCarContinuous-v0
tmp_env = gym.make('BreakoutNoFrameskip-v4')
# num_actions = tmp_env.action_space.n
print(tmp_env.observation_space.shape, tmp_env.action_space )
print(isinstance(tmp_env.action_space, gym.spaces.Box))
print(list((1, 2)))

def func1(a,b):
    def func2(c, d):
        return c*d - a*b
    return func2


f2 = func1(2, 3)

res = f2(5, 6)

print(res)