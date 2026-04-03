import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# === Функция инвертирования доски: сеть всегда видит себя как 1 (X) ===
def invert_board(board, player):
    """Если игрок 2 (O), то 1 ↔ 2, чтобы сеть всегда играла за X"""
    board = np.array(board)
    if player == 2:
        board = np.where(board == 1, 2, np.where(board == 2, 1, 0))
    return board


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Node:
    def __init__(self, state, player=2, parent=None, prior=0.0):
        self.state = np.array(state)
        self.player = player        # 2 — O (AZ), 1 — X
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def is_terminal(self):
        return self.check_winner() != 3

    def check_winner(self):
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for a, b, c in lines:
            if self.state[a] == self.state[b] == self.state[c] == 1:
                return 1
            if self.state[a] == self.state[b] == self.state[c] == 2:
                return 2
        return 0 if 0 not in self.state else 3

    def value(self):
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def expand(self, policy_logits):
        mask = (self.state == 0)
        logits = policy_logits.copy()
        logits[~mask] = -np.inf
        probs = np.exp(logits - np.max(logits))  # stable softmax
        probs[~mask] = 0
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(9)/9

        for action in range(9):
            if self.state[action] == 0:
                new_state = self.state.copy()
                new_state[action] = self.player
                next_player = 1 if self.player == 2 else 2
                self.children[action] = Node(new_state, next_player, self, probs[action])
        self.is_expanded = True

    def select(self, c=2.5):
        return max(self.children.values(), key=lambda n: n.puct(c))

    def puct(self, c=2.5):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        U = c * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        Q = self.value()
        return Q + U

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)


class AZAgent:
    def __init__(self):
        self.policy = PolicyNet()
        self.value = ValueNet()
        self.opt_p = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.opt_v = optim.Adam(self.value.parameters(), lr=1e-3)
        self.memoryP = deque(maxlen=1000)  # политики
        self.memoryV = deque(maxlen=1000)  # ценности

    def get_policy_value(self, state, player=2):
        state = np.array(state)
        if player == 1:  # если настоящий игрок — X
            inv_state = np.where(state == 1, 2, np.where(state == 2, 1, 0))
        else:  # player == 2 → O → сеть и так за O → не трогаем
            inv_state = state.copy()

        with torch.no_grad():
            s = torch.FloatTensor(inv_state).unsqueeze(0)
            p_logits = self.policy(s).squeeze().numpy()
            v = self.value(s).item()

        return p_logits, v

    def compute_root_value(self, root):
        """Вычисляем v_target как взвешенное среднее Q по детям"""
        if not root.children:
            winner = root.check_winner()
            if winner == 0:
                return 0.0
            return +1.0 if winner == root.player else -1.0

        total_visits = 0
        weighted_sum = 0.0
        for child in root.children.values():
            if child.visits > 0:
                weighted_sum += child.value() * child.visits
                total_visits += child.visits
        return weighted_sum / total_visits if total_visits > 0 else 0.0

    def act(self, state, sims=100):
        root = Node(state, player=2)

        for _ in range(sims):
            node = root
            path = []

            # Selection
            while node.children and not node.is_terminal():
                node = node.select()
                path.append(node)

            # Evaluation
            if node.is_terminal():
                winner = node.check_winner()
                if winner == 0:
                    value = 0.0
                elif winner == node.player:
                    value = +1.0
                else:
                    value = -1.0
            else:
                p_logits, v = self.get_policy_value(node.state, node.player)
                node.expand(p_logits)
                value = v

            # Backpropagate по всему пути
            node.backpropagate(-value)


        v_target = self.compute_root_value(root)
        # Политика: посещения
        visits = [root.children[a].visits if a in root.children else 0 for a in range(9)]
        total_visits = sum(visits)
        pi = [v / total_visits for v in visits] if total_visits > 0 else [1/9]*9


        self.memoryP.append((np.array(state), pi))
        self.memoryV.append((np.array(state), v_target))

        # Выбор действия
        action = np.argmax(visits)


        self.train_nets(32,1)

        return action


    def train_nets(self, batch_size=30, epochs=1):
        """Обучаем на случайных батчах"""
        if len(self.memoryP) < batch_size or len(self.memoryV) < batch_size:
            return

        self.policy.train()
        self.value.train()

        for _ in range(epochs):
            # === Обучение политики ===
            batch_p = random.sample(self.memoryP, batch_size)
            states_p = np.array([s for s, _ in batch_p])  # AZ — player 2
            target_pis = np.array([pi for _, pi in batch_p])

            s_p = torch.FloatTensor(states_p)
            pi_t = torch.FloatTensor(target_pis)
            log_probs = torch.log_softmax(self.policy(s_p), dim=1)
            loss_p = -(pi_t * log_probs).sum(dim=1).mean()

            self.opt_p.zero_grad()
            loss_p.backward()
            self.opt_p.step()

            # === Обучение ценности ===
            batch_v = random.sample(self.memoryV, batch_size)
            states_v = np.array([invert_board(s, 2) for s, _ in batch_v])
            target_zs = torch.FloatTensor([z for _, z in batch_v]).unsqueeze(1)

            s_v = torch.FloatTensor(states_v)
            v_pred = self.value(s_v)
            loss_v = torch.nn.functional.mse_loss(v_pred, target_zs)

            self.opt_v.zero_grad()
            loss_v.backward()
            self.opt_v.step()
