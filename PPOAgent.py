import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque



class Crtic(nn.Module):
    def __init__(self):
        super(Crtic, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = torch.nn.ReLU()
        self.sp = torch.nn.Softplus()
    def forward(self, x):
        x = self.sp(self.fc1(x))
        x = self.sp(self.fc2(x))
        x = self.sp(self.fc3(x))
        probs = torch.softmax(x, dim=-1)
        return probs


class PPOAgent:
    def __init__(self):
        self.modelA = Actor()
        self.modelC = Crtic()
        self.optimizerA = optim.SGD(self.modelA.parameters(), lr=0.0002)
        self.optimizerC = optim.SGD(self.modelC.parameters(), lr=0.002)
        self.curr_prob = 0
        self.curr_trac = list()
        self.trajcs = list()
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.modelA(state_tensor)

        if torch.isnan(probs).any():
            valid_actions = [i for i, cell in enumerate(state) if cell == 0]
            action = random.choice(valid_actions)
            self.curr_prob = np.ones(9)/9.0
            return action
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
        if len(self.trajcs) > 50:
            self.trajcs.clear()

    def learn(self):

        if len(self.trajcs) < 20:
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

            Rlist = list()

            for epoch in range(1):
                for line in data:
                    state, act, prob, R = line
                    Rlist.append(R)
                    state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
                    Rew = torch.tensor(R,dtype=torch.float).reshape([1,1])

                    V = self.modelC(state)
                    loss = nn.MSELoss()(V, Rew)
                    self.optimizerC.zero_grad()
                    loss.backward()
                    self.optimizerC.step()

            baseline = np.mean(Rlist)

            for epoch in range(3):
                for line in data:
                    state, act, oldProbs, R = line
                    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                    Rew = torch.tensor(R, dtype=torch.float).unsqueeze(0)
                    oldProbs = torch.tensor(oldProbs, dtype=torch.float).unsqueeze(0)

                    with torch.no_grad():
                        V = self.modelC(state)
                    A = (Rew-baseline).detach().item()


                    probs = self.modelA(state)[0][act]

                    ratio = torch.exp(torch.log(probs) - torch.log(oldProbs))
                    loss = -torch.min(A*ratio, torch.clamp(ratio, 0.8, 1.2))
                    self.optimizerA.zero_grad()
                    loss.backward()
                    self.optimizerA.step()

        self.trajcs.clear()
