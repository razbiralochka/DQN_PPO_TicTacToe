import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.l1 = nn.Linear(9,16)
        self.l2 = nn.Linear(16,16)
        self.l3 = nn.Linear(16,9)
        self.relu = nn.ReLU()
        self.sp = torch.nn.Softplus()
    def forward(self, x):
        x = self.sp(self.l1(x))
        x = self.sp(self.l2(x))
        x = self.l3(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 18)
        self.fc2 = nn.Linear(18, 1)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()  # Нормализация в [-1, 1]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.tanh(x)  # Выход в диапазоне [-1, 1]


class AZAgent:
    def __init__(self):
        self.enemy = Policy()
        self.policy = Policy()
        self.value = ValueNetwork()
        self.optimizerE = optim.Adam(self.enemy.parameters(), lr=0.001)
        self.optimizerP = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizerV = optim.Adam(self.value.parameters(), lr=0.005)
        self.enemyData = deque(maxlen=200)
        self.policyData = deque(maxlen=200)
        self.valueData = deque(maxlen=1000)


    def watchEnemy(self, state, action):
        self.enemyData.append((state,action))
        self.train_enemy_model()

    def trainValue(self):

        if len(self.valueData) < 50:
            return

        batch = random.sample(self.valueData, 50)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float)
        z_val = torch.tensor([b[1] for b in batch], dtype=torch.float).unsqueeze(1)
        self.optimizerV.zero_grad()
        predicted_val = self.value(states)
        loss = nn.MSELoss()(predicted_val, z_val)
        loss.backward()
        self.optimizerV.step()
        #print(loss.item())


    def trainSelf(self, state, action):

        self.policyData.append((state,action))
        if len(self.policyData) < 20:
            return
        batch = random.sample(self.policyData, 20)
        states = torch.tensor([line[0] for line in batch], dtype=torch.float)
        actions = torch.tensor([line[1] for line in batch], dtype=torch.long)
        self.optimizerP.zero_grad()
        predicted_probs = self.policy(states)
        loss = nn.CrossEntropyLoss()(predicted_probs, actions)
        loss.backward()
        self.optimizerP.step()

    def train_enemy_model(self):

        if len(self.enemyData) < 20:
            return

        batch = random.sample(self.enemyData, 20)

        states = torch.tensor([line[0] for line in batch], dtype=torch.float)
        actions = torch.tensor([line[1] for line in batch], dtype=torch.long)

        self.optimizerE.zero_grad()
        predicted_probs = self.enemy(states)
        loss = nn.CrossEntropyLoss()(predicted_probs, actions)
        loss.backward()
        self.optimizerE.step()



    def checkState(self, state):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Горизонтали
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Вертикали
            [0, 4, 8], [2, 4, 6]]  # Диагонали

        for combo in winning_combinations:
            if state[combo[0]] == state[combo[1]] == state[combo[2]] == 1:
                 return 1  # Победа крестиков

        for combo in winning_combinations:
            if state[combo[0]] == state[combo[1]] == state[combo[2]] == 2:
                return 2  # Победа ноликов

        if 0 not in state:
                return 0  # Ничья

                # Игра продолжается (3)
        return 3


    def act(self,state, sims, maxDepth):
        available = [i for i, x in enumerate(state) if x == 0]
        if not available:
            return 0
        #dynsims = round(sims * (len(available)/8)**2)
        with torch.no_grad():
            logits = self.policy(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            prior_probs = torch.softmax(logits, dim=1).numpy().flatten()

        N = np.zeros(len(available))
        totalR = np.zeros(len(available))
        U = np.zeros(len(available))

        for sim in range(sims):

            for i in range(len(available)):
                Q = totalR[i]/(N[i]+1)
                U[i] = Q +  prior_probs[available[i]]*np.sqrt(np.sum(N)) / (1 + N[i])
            idx = np.argmax(U)
            action = available[idx]

            simState = state.copy()
            simState[action] = 2
            depth = 0
            while self.checkState(simState) == 3 and depth < maxDepth:
                enemyAct = self.predict_enemy_move(simState)

                simState[enemyAct] = 1

                if self.checkState(simState) != 3:
                    break

                selfAct = self.predict_self_move(simState)

                simState[selfAct] = 2
                depth = depth + 1

            R = 0
            winner = self.checkState(simState)
            if winner != 3:
                R = 1 if winner == 2 else -1 if winner == 1 else 0
            else:
                with torch.no_grad():
                    stateT = torch.tensor(simState, dtype=torch.float).unsqueeze(0)
                    V = self.value(stateT)
                    R = V.item()

            self.valueData.append((state, R))
            totalR[idx] += R

            N[idx] = N[idx]+1

        best_idx = np.argmax(N)
        best_action = available[best_idx]
        self.trainSelf(state,best_action)
        return best_action

    def predict_enemy_move(self, state):
        available = [i for i, x in enumerate(state) if x == 0]
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            logits = self.enemy(state_tensor).squeeze()
            masked_logits = np.full(9, -np.inf)
            masked_logits[available] = logits[available].numpy()
            probs = torch.softmax(torch.tensor(masked_logits), dim=0)
        action_dist = torch.distributions.Categorical(probs=probs)
        return action_dist.sample().item()

    def predict_self_move(self, state):
        available = [i for i, x in enumerate(state) if x == 0]
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_tensor).squeeze()
            masked_logits = np.full(9, -np.inf)
            masked_logits[available] = logits[available].numpy()
            probs = torch.softmax(torch.tensor(masked_logits), dim=0)
        action_dist = torch.distributions.Categorical(probs=probs)
        return action_dist.sample().item()

    def checkNet(self, state, reward):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        R = torch.tensor(-reward,dtype=torch.float)
        predicted_val = self.value(state_tensor)
        loss = nn.MSELoss()(predicted_val,R)
        print(loss.item())