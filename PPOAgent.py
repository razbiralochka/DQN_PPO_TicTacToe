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
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.tanh(x)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = torch.nn.LeakyReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        probs = torch.softmax(x, dim=-1)
        return probs


class PPOAgent:
    def __init__(self):
        self.modelA = Actor()
        self.modelC = Crtic()
        self.optimizerA = optim.Adam(self.modelA.parameters(), lr=1e-3)
        self.optimizerC = optim.Adam(self.modelC.parameters(), lr=1e-3)
        self.curr_prob = 0
        self.curr_trac = list()
        self.trajcs = deque(maxlen=15)
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.modelA(state_tensor)

        if torch.isnan(probs).any():
            valid_actions = [i for i, cell in enumerate(state) if cell == 0]
            action = random.choice(valid_actions)
            self.curr_prob = np.ones(9)/9.0
            self.modelA = Actor()
            print("NAN")
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

    def rememberTraj(self, finalR):
        for step in self.curr_trac:
           step[3] = finalR

        self.trajcs.append(self.curr_trac.copy())
        self.curr_trac.clear()


    def learn(self):

        if len(self.trajcs) < 15:
            return
        data = list()
        for traj in self.trajcs:
            for step in traj:  # ← проходим по каждому шагу в траектории
                data.append(step)
        self.trajcs.clear()
        #print(data)

        random.shuffle(data)


        for epoch in range(50):
            for line in data:
                state, act, prob, R = line
                stateT = torch.tensor(state,dtype=torch.float)
                Rew = torch.tensor(R,dtype=torch.float).unsqueeze(0)

                V = self.modelC(stateT)


                loss = nn.MSELoss()(V, Rew)
                self.optimizerC.zero_grad()
                loss.backward()
                self.optimizerC.step()


        for epoch in range(15):

            for line in data:
                state, act, oldProbs, R = line
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                Rew = torch.tensor(R, dtype=torch.float).unsqueeze(0)
                oldProbs = torch.tensor(oldProbs, dtype=torch.float).unsqueeze(0)

                with torch.no_grad():
                    V = self.modelC(state)
                A = (Rew-V).detach()

                self.optimizerA.zero_grad()
                probs = self.modelA(state)[0][act]

                ratio = probs /(oldProbs + 1e-8)
                loss = -torch.min(A*ratio, A*torch.clamp(ratio, 0.8, 1.2)).mean()

                loss.backward()
                self.optimizerA.step()


