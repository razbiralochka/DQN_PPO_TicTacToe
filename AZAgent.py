import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F

# === Нейронные сети ===
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(9, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 9)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Без ReLU перед Tanh
        return self.tanh(x)

# === Узел MCTS ===
class Node:
    def __init__(self, state, parent=None, prior=1.0, player=1):
        self.state = np.array(state)
        self.parent = parent
        self.player = player  # 1 — X, 2 — O
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def check_state(self):
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for a,b,c in wins:
            if self.state[a] == self.state[b] == self.state[c] == 1:
                return 1
            if self.state[a] == self.state[b] == self.state[c] == 2:
                return 2
        return 0 if 0 not in self.state else 3  # 3 = игра идёт

    def is_terminal(self):
        return self.check_state() != 3

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        available = [i for i, x in enumerate(self.state) if x == 0]
        return len(self.children) == len(available)

    def select(self, c=1.5):
        unvisited = [child for child in self.children.values() if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        return max(self.children.values(), key=lambda child: child.puct_score(c))

    def puct_score(self, c=1.5):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        return self.value() + exploration

    def expand(self, policy_logits, player):
        policy_probs = torch.softmax(torch.tensor(policy_logits), dim=0).numpy()
        available = [i for i, x in enumerate(self.state) if x == 0]
        for action in available:
            new_state = self.state.copy()
            new_state[action] = player
            next_player = 1 if player == 2 else 2
            self.children[action] = Node(
                state=new_state,
                parent=self,
                prior=policy_probs[action],
                player=next_player
            )

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)  # инвертируем для противника

# === Агент AlphaZero ===
class AZAgent:
    def __init__(self):
        self.policy = Policy()
        self.value = ValueNetwork()
        self.optimizerP = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizerV = optim.Adam(self.value.parameters(), lr=0.005)
        self.policyData = deque(maxlen=5000)
        self.valueData = deque(maxlen=5000)

    def trainValue(self):
        """Обучение Value Network на статистике MCTS"""
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

    def trainSelf(self, state, target_policy):
        """Обучение Policy Network"""
        self.policyData.append((state, target_policy))
        if len(self.policyData) < 20:
            return
        batch = random.sample(self.policyData, 20)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float)
        self.optimizerP.zero_grad()
        logits = self.policy(states)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, targets, reduction='batchmean')
        loss.backward()
        self.optimizerP.step()

    def act(self, state, player, sims=800, temperature=1.0):
        """Выбор хода через MCTS с сохранением данных для обучения"""
        root = Node(state, player=player)

        # MCTS симуляции
        for _ in range(sims):
            node = root

            # Спуск до листа ИЛИ нерасширенного узла
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select(c=1.5)

            # Расширение, если узел не терминал
            if not node.is_terminal():
                with torch.no_grad():
                    state_tensor = torch.tensor(node.state, dtype=torch.float).unsqueeze(0)
                    policy_logits = self.policy(state_tensor).squeeze().numpy()
                node.expand(policy_logits, node.player)

                # Оценка листа
                if node.is_terminal():
                    winner = node.check_state()
                    value = 1.0 if winner == 2 else (-1.0 if winner == 1 else 0.0)
                else:
                    with torch.no_grad():
                v = self.value(state_tensor).item()
            value = v
                node.backpropagate(value)

        # Сбор данных для обучения Value Network
        for action, child in root.children.items():
            if child.visits > 0:
                # Оценка позиции из статистики MCTS
                node_value = child