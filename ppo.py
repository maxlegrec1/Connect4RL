import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import wandb

# Assuming Connect4 class is implemented similarly to the original code
from connect4 import Connect4

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 7)  # 7 possible actions in Connect4
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1, 7, 7)  # Reshape to (batch_size, channels, height, width)
        x = self.conv(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

class PPOAgent:
    def __init__(self, learning_rate, gamma, clip_epsilon, n_epochs, batch_size):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.policy = Connect4Net()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.memory = PPOMemory()

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs, value = self.policy(state)
        action_probs = Categorical(probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches(self.batch_size)

            values = torch.tensor(vals_arr)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*0.95
                advantage[t] = a_t
            advantage = torch.tensor(advantage)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(old_prob_arr[batch])
                actions = torch.tensor(action_arr[batch])

                new_probs, critic_value = self.policy(states)
                critic_value = critic_value.squeeze()

                new_probs = Categorical(new_probs)
                new_log_probs = new_probs.log_prob(actions)
                prob_ratio = (new_log_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()

def train():
    agent = PPOAgent(learning_rate=0.0003, gamma=0.99, clip_epsilon=0.2, n_epochs=4, batch_size=64)
    n_games = 10000
    
    wandb.init(project="connect4-ppo")

    for i in tqdm(range(n_games)):
        env = Connect4()
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            new_observation, reward, done, _ = env.step(action)
            score += reward
            agent.memory.store_memory(observation, action, prob, val, reward, done)
            observation = new_observation
            
        agent.learn()
        
        wandb.log({
            "game": i,
            "score": score
        })
        
        if i % 10 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_model_{i}.pt")

if __name__ == "__main__":
    train()