import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

lr = 0.1
discount = 0.95
episodes = 25000
show_every = 2000
epsilon = 0.1  # Exploration rate

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(episodes):
    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        # Exploration-exploitation trade-off
        if np.random.random() < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - lr) * current_q + lr * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
            print(f"we made it in {episode}")

        discrete_state = new_discrete_state

env.close()
