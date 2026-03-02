from auxFunctions import getState, load_obj, maxAction
import gym

env = gym.make('MountainCar-v0')
env._max_episode_steps = 200

Q = load_obj('Q-table-Q-Learning')

total_success = 0
total_steps = 0
total_return = 0
eval_episodes = 100

for episode in range(eval_episodes):
    observation = env.reset()
    state = getState(observation)
    done = False

    steps = 0
    episode_return = 0

    print("Episode #", episode)

    while not done:
        # env.render()  # optional
        # print(observation)  # optional

        action = int(maxAction(Q, state))   # ensure int action
        observation, reward, done, info = env.step(action)

        episode_return += reward
        steps += 1
        state = getState(observation)

    final_pos = observation[0]
    reached_goal = final_pos >= 0.5

    print(f"Episode {episode}: steps={steps}, return={episode_return}, "
          f"final_pos={final_pos:.3f}, goal={reached_goal}")

    if reached_goal:
        total_success += 1

    total_steps += steps
    total_return += episode_return

env.close()

success_rate = total_success / eval_episodes
avg_steps = total_steps / eval_episodes
avg_return = total_return / eval_episodes

print("\n=== Evaluation Summary (Q-learning) ===")
print("Success Rate:", success_rate)
print("Average Steps:", avg_steps)
print("Average Return:", avg_return)