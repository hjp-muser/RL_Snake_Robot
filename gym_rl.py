import gym
import rlbench.gym_wrapper

env = gym.make('reach_target-vision-v0')
training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    print('--------------------------------------------')
    # env.render()  # Note: rendering increases step time.

print('Done')
env.close()
