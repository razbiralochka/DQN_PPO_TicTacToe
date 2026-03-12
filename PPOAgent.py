import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque



class Crtic(nn.Module):
    def __init__(self):
        super(Crtic, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.fc3(x), dim=-1)
        return probs


class PPOAgent:
    def __init__(self):
        self.modelA = Actor()
        self.modelC = Crtic()
        self.optimizerA = optim.SGD(self.modelA.parameters(), lr=0.0005)
        self.optimizerC = optim.SGD(self.modelC.parameters(), lr=0.002)
        self.curr_prob = 0
        self.curr_trac = list()
        self.trajcs = list()
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.modelA(state_tensor)
        self.curr_prob = probs.detach().numpy()[0]
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        #action = torch.argmax(probs)
        return action.item()

    def remember(self, state_, a, r):
        old_prob = self.curr_prob[a]
        reward = r
        state = state_
        self.curr_trac.append([state,a,old_prob,reward])

    def rememberTraj(self):
        self.trajcs.append(self.curr_trac.copy())
        self.curr_trac.clear()

    def learn(self):

        if len(self.trajcs) < 25:
            return

        data = list()
        for traj in self.trajcs:
            L = len(traj)
            for i in reversed(range(L)):

                if i < L-1:
                    traj[i][3] = traj[i][3] + traj[i+1][3]

            for elem in traj:
                data.append(elem)
            random.shuffle(data)
            for epoch in range(100):
                for line in data:
                    state, act, prob, R = line
                    state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
                    Rew = torch.tensor(R,dtype=torch.float32).reshape([1,1])

                    V = self.modelC(state)
                    loss = nn.MSELoss()(V, Rew)
                    self.optimizerC.zero_grad()
                    loss.backward()
                    self.optimizerC.step()

            for epoch in range(10):
                for line in data:
                    state, act, oldProbs, R = line
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    Rew = torch.tensor(R, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        V = self.modelC(state)
                    A = (Rew-0.0*V).detach().item()

                    #print(A, Rew, V)
                    probs = self.modelA(state)[0][act]
                    #print(probs, oldProbs)
                    loss = -torch.min(A*probs/oldProbs, torch.clamp(probs/oldProbs, 0.2, 1.2))
                    self.optimizerA.zero_grad()
                    loss.backward()
                    self.optimizerA.step()

        self.trajcs.clear()
