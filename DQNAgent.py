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

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=100)
        self.model = DQN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0002)

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)

        if random.randint(0,1000) < 200:
            return random.randint(0,8)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < 10:
            return
        minibatch = random.sample(self.memory, 10)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            q_values = self.model(state)
            target = q_values.clone()

            with torch.no_grad():
                qNext = self.model(next_state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + torch.max(qNext)

            loss = nn.MSELoss()(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()