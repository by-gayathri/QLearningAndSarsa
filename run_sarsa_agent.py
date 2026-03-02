from auxFunctions import getState, load_obj, maxAction
import gym
import numpy as np

env = gym.make('MountainCar-v0')
env._max_episode_steps = 200

# Load trained SARSA Q-table
Q = load_obj('pre-trained-SARSA')

eval_episodes = 100

total_success = 0
total_steps = 0
total_return = 0

for episode in range(eval_episodes):

    observation = env.reset()
    state = getState(observation)
    done = False

    steps = 0
    episode_return = 0

    while not done:
        # Pure greedy action (no exploration)
        action = int(maxAction(Q, state))

        observation, reward, done, info = env.step(action)

        episode_return += reward
        steps += 1
        state = getState(observation)

    # Check if goal reached
    if observation[0] >= 0.5:
        total_success += 1

    total_steps += steps
    total_return += episode_return

env.close()

# Compute metrics
success_rate = total_success / eval_episodes
avg_steps = total_steps / eval_episodes
avg_return = total_return / eval_episodes

print("Success Rate:", success_rate)
print("Average Steps:", avg_steps)
print("Average Return:", avg_return)