import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = torch.nn.ReLU()
        self.sp = torch.nn.Softplus()
    def forward(self, x):
        x = self.sp(self.fc1(x))
        x = self.sp(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)

        q_values = q_values.detach()[0].tolist()
        valid_actions = [i for i, cell in enumerate(state) if cell == 0]
        valid_q_values = [q_values[i] for i in valid_actions]
        best_idx = np.argmax(valid_q_values)
        action = valid_actions[best_idx]

        #if random.uniform(0,100) < 5:
            # = random.choice(valid_actions)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < 20:
            return
        minibatch = random.sample(self.memory, 20)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            q_values = self.model(state)
            target = q_values.clone()

            self.optimizer.zero_grad()

            with torch.no_grad():
                qNext = self.model(next_state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + torch.max(qNext)

            loss = nn.MSELoss()(q_values, target)

            loss.backward()
            self.optimizer.step()