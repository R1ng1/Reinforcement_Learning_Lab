import time
import gymnasium as gym
from traffic_environment import TrafficEnv
import rl_planners
import numpy as np

# define rewards function
rewards = {"state": 0}

# initialize the environment with rewards and max_steps
env = TrafficEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm to plan or train an agent
rl_algo = "Value Iteration"

# initialize the agent and train it
if rl_algo == "Value Iteration":
    agent = rl_planners.ValueIterationPlanner(env)
elif rl_algo == "Policy Iteration":
    agent = rl_planners.PolicyIterationPlanner(env)


# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Count of instances where car count exceeds critical thresholds (N total cars or M in any direction)
# 2. Average number of cars waiting at the intersection in all directions during a time period
# 3. Maximum continuous time where car count remains below critical thresholds
critical_count = 0
CRITICAL_THRESHOLD = 10

total_cars_waiting = 0
step_count = 0

current_safe_duration = 0
max_safe_duration = 0

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}
np.random.seed(42)
env.action_space.seed(42)

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.
cumulative_reward = 0
episode_steps = 0
# set light state variables
RED, GREEN = 0, 1

# run the environment until terminated or truncated
terminated, truncated = False, False
while (not terminated and not truncated):

    action = agent.choose_action(observation)

    observation, reward, terminated, truncated, info = env.step(action)

    # unpack observation
    ns, ew, light = tuple(observation)

    # update cumulative reward
    cumulative_reward += reward

    # count steps
    step_count += 1
    episode_steps += 1
    total_throughput = 0

    # update total cars waiting
    total_cars_waiting += ns + ew

    # check critical condition
    if ns > CRITICAL_THRESHOLD or ew > CRITICAL_THRESHOLD:
        critical_count += 1
        current_safe_duration = 0
    else:
        current_safe_duration += 1
        max_safe_duration = max(max_safe_duration, current_safe_duration)

    light_color = "GREEN" if light == GREEN else "RED"

    print(f"Step: {episode_steps}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}")

    env.render()

    time.sleep(0.9)
    # reset the environment if terminated or truncated
    if terminated or truncated:
        print("\nTERMINATED OR TRUNCATED, RESETTING...\n")

        # TODO: Update metrics for completed episode
        print("Episode Steps:", episode_steps)
        print("Episode Reward:", cumulative_reward)

        observation, info = env.reset(), {}

        # TODO: Reset tracking variables for the new episode
        episode_steps = 0
        cumulative_reward = 0
        current_safe_duration = 0

        terminated, truncated = False, False

# close the environment
env.render(close=True)

# TODO: Evaluate performance based on high-level metrics
average_cars_waiting = total_cars_waiting / step_count
# TODO: Print performance metrics
print(f"Critical Traffic Instances: {critical_count}")
print(f"Average Cars Waiting: {average_cars_waiting:.2f}")
print(f"Maximum Safe Duration: {max_safe_duration}")
print("\n=== PERFORMANCE EVALUATION ===")
