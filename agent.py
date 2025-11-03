import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class HangmanDQNAgent:
    def __init__(self, max_word_len=20, alpha=0.0005, gamma=0.95, epsilon_start=0.8, epsilon_min=0.15, epsilon_decay=0.999, batch_size=128):
        self.max_word_len = max_word_len
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_dim = 2 * max_word_len + 26 + 26 + 1
        self.output_dim = 26
        self.model = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_model = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.memory = deque(maxlen=20000)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.letter_to_idx = {chr(97 + i): i for i in range(26)}
        self.idx_to_letter = {i: chr(97 + i) for i in range(26)}
        
        self.step_count = 0
        self.update_target_freq = 2000

    def get_state_vector(self, masked_word, guessed_letters, hmm_probs, lives_left):
        mask_binary = [1 if c != '_' else 0 for c in masked_word]
        if len(mask_binary) > self.max_word_len:
            mask_binary = mask_binary[:self.max_word_len]
        else:
            mask_binary = mask_binary + [0] * (self.max_word_len - len(mask_binary))

        mask_chars = [self.letter_to_idx.get(c, 0) / 26.0 if c != '_' else 0 for c in masked_word]
        if len(mask_chars) > self.max_word_len:
            mask_chars = mask_chars[:self.max_word_len]
        else:
            mask_chars = mask_chars + [0] * (self.max_word_len - len(mask_chars))
        guessed_bin = [1 if self.idx_to_letter[i] in guessed_letters else 0 for i in range(26)]
        probs = hmm_probs / (hmm_probs.sum() + 1e-12)
        lives = [lives_left / 6.0]
        state = np.concatenate([mask_binary, mask_chars, guessed_bin, probs, lives])
        return torch.FloatTensor(state).to(self.device)

    def choose_action(self, state, guessed_letters, hmm_probs):
        available_actions = [i for i in range(26) if self.idx_to_letter[i] not in guessed_letters]
        if not available_actions:
            return None
        
        if random.random() < self.epsilon:
            avail_probs = np.array([hmm_probs[i] for i in available_actions])
            avail_probs = avail_probs / (avail_probs.sum() + 1e-12)
            return np.random.choice(available_actions, p=avail_probs)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                masked_q = torch.full((26,), float('-inf')).to(self.device)
                for a in available_actions:
                    masked_q[a] = q_values[a]
                return torch.argmax(masked_q).item()

    def update(self, state, action, reward, next_state, done, next_available_actions):
        if reward == -10:  
            reward *= 2 
        self.memory.append((state, action, reward, next_state, done, next_available_actions))
        self.step_count += 1
        
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_avails = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_q_values = torch.zeros(self.batch_size).to(self.device)
        for i in range(self.batch_size):
            if not dones[i]:
                avail = next_avails[i]
                if avail:
                    with torch.no_grad():
                        next_qs = self.target_model(next_states[i])
                        next_q_values[i] = max(next_qs[a] for a in avail)
        
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="models/dqn_agent.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path="models/dqn_agent.pt"):
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state)
            except Exception as e:
                print(f"Warning: failed to load weights strictly due to: {e}. Trying non-strict...")
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            print("No saved DQN model found. Starting fresh.")


