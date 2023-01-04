"""
代码来自https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb
稍微有改动，因为gym的api变了。
"""

import numpy as np
import gym
import random


env = gym.make("Taxi-v3", render_mode="rgb_array")
env.reset()
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)


# 设置超参
total_episodes = 1000  # Total episodes
total_test_episodes = 100  # Total test episodes
max_steps = 99  # Max steps per episode

learning_rate = 0.7  # Learning rate
gamma = 0.618  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob



"""
q-learning主体方法
"""
# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()[0]
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, *abs = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                                                         np.max(qtable[new_state, :]) - qtable[
                                                                             state, action])

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

"""
训练结束，测试。
"""

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    state = state[0]
    step = 0
    done = False
    total_rewards = 0
    # print("****************************************************")
    # print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, *abc = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print ("Score", total_rewards)
            break
        state = new_state
print("训练结束后 qtable:\n",qtable)
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))
