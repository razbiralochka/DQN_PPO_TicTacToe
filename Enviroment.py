import random

class TicTacToeEnv:
    def __init__(self):
        self.board = [0] * 9  # 0 - пустая клетка, 1 - крестик, 2 - нолик

    def reset(self):
        self.board = [0] * 9

    def getState(self):
        return self.board.copy()

    def step(self, action, player):
        if self.board[action] != 0:
            available_actions = [i for i, x in enumerate(self.board) if x == 0]



            action = random.choice(available_actions)

        self.board[action] = player
        gameState = self.checkBoard()

        return action, gameState


    def checkBoard(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Горизонтали
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Вертикали
            [0, 4, 8], [2, 4, 6]]  # Диагонали


        # Проверка на победу крестиков (1)
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] == 1:
                 return 1  # Победа крестиков

        # Проверка на победу ноликов (2)
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] == 2:
                return 2  # Победа ноликов

        # Проверка на ничью (0)
        if 0 not in self.board:
                return 0  # Ничья

        # Игра продолжается (3)
        return 3

