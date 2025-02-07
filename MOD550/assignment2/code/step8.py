import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# GridWorld Environment
class GridWorld:
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        
        self.state = (x, y)
        
        if self.state in self.obstacles:
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        self.state = (0, 0)
        return self.state

# Q-Learning Agent
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=100):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        rewards, states, starts, steps_per_episode, q_value_changes = [], [], [], [], []
        prev_q_table = np.copy(self.q_table)
        
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps_in_episode = 0

            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)
                steps_in_episode += 1

            # Compute Q-value change magnitude
            q_change = np.abs(self.q_table - prev_q_table).sum()
            q_value_changes.append(q_change)
            prev_q_table = np.copy(self.q_table)

            # Check if the mean of the last 15 Q-value changes drops below 0.2
            if len(q_value_changes) >= 15 and np.mean(q_value_changes[-15:]) < 0.2:
                return episode  # Return the number of episodes needed to converge
            
            starts.append(len(states))
            rewards.append(total_reward)
            steps_per_episode.append(steps_in_episode)

        return rewards, self.episodes, steps_per_episode, q_value_changes, states, starts

# Run Environment and Agent
env = GridWorld(size=5, num_obstacles=5)
agent = QLearning(env, episodes=100)

# Train Agent
train_result = agent.train()

# Ensure train_result is always unpackable
if isinstance(train_result, int):  # If convergence happened early, wrap it in a tuple
    train_result = ([], train_result, [], [], [], [])

rewards, total_episodes, steps_per_episode, q_value_changes, states, starts = train_result

# --------------------------
# Run Experiments for Different Learning Rates
# --------------------------
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]  # Different alphas to test
convergence_episodes = []

for alpha in learning_rates:
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env, alpha=alpha, episodes=100)  # Set a max of 100 episodes
    print(f"Testing learning rate: {alpha}")
    
    episodes_needed = agent.train()

    if isinstance(episodes_needed, list):  
        episodes_needed = episodes_needed[0] if len(episodes_needed) > 0 else 100

    episodes_needed = int(episodes_needed)

    convergence_episodes.append(episodes_needed)
    print(f"Converged in {episodes_needed} episodes\n")

# --------------------------
# Plot Learning Rate vs Convergence
# --------------------------
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, convergence_episodes, marker='o', linestyle='-', color='blue')
plt.xlabel("Learning Rate (alpha)")
plt.ylabel("Episodes to Converge")
plt.title("Convergence vs. Learning Rate")
plt.grid(True)
plt.savefig("step8_learning_rate_vs_convergence.png")
plt.show()
