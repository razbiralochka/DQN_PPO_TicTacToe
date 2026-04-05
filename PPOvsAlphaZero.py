import numpy as np
from AZAgent import AZAgent
from Environment import TicTacToeEnv
from PPOAgent import PPOAgent
import matplotlib.pyplot as plt
import random


env = TicTacToeEnv()


alphaZero = AZAgent()
ppoA = PPOAgent()


sims = 0
S = 0
Games = 0
Score = list()


for episode in range(1000):
    env.reset()
    state2 = None  # предыдущее состояние DQN
    act0 = None    # предыдущее действие DQN
    game_over = False

    while not game_over:
        # Ход PPO (крестики, игрок 1)
        state1 = env.getState()
        actX = ppoA.act(state1)
        actX, stat = env.step(actX, 1)  # stat — текущий статус игры: 0, 1, 2, 3
        ppo_reward = 0

        # Проверяем, закончилась ли игра после хода PPO
        result = env.checkBoard()
        if result != 3:
            game_over = True
            if result == 1:
                ppo_reward = 1
            elif result == 2:
                ppo_reward = -1
            else:
                ppo_reward = 0

            # DQN проиграл, если PPO выиграл, и наоборот
            az_reward = -ppo_reward
        else:
            ppo_reward = 0
            az_reward = 0  # игра продолжается



        # Если игра не окончена — DQN делает ход
        if not game_over:
            state2 = env.getState()
            act0 = alphaZero.act(state2, sims =200)
            act0, stat = env.step(act0, 2)

            # Проверяем, закончилась ли игра после хода DQN
            result = env.checkBoard()
            if result != 3:
                game_over = True
                if result == 2:
                    az_reward = 1
                elif result == 1:
                    az_reward = -1
                else:
                    az_reward = 0
                ppo_reward = -az_reward
            else:
                az_reward = 0
                ppo_reward = 0

            # Обновляем PPO: его действие, награда, закончилась ли игра
            ppoA.remember(state1, actX, ppo_reward)

        # Обучение
        final_reward = 1 if env.checkBoard() == 1 else (-1 if env.checkBoard() == 2 else 0)
        ppoA.rememberTraj(final_reward)

    # После завершения игры
    ppoA.learn()

    alphaZero.memoryV.append((env.getState(), 1.0 if env.checkBoard() == 2 else -1.0))
    _, v = alphaZero.get_policy_value(env.getState(), 2)
    # Обновляем счёт
    result = env.checkBoard()
    Games += 1
    if result == 1:
        S += 1
        print(Games, env.board, ' CrossWin ', S, v)
    elif result == 2:
        S -= 1
        print(Games, env.board, ' ZeroWin ', S, v)
    else:
        print(Games, env.board, ' Draw ', S, v)
    Score.append(S)

    if episode % 100 == 0:
        np.savetxt('Stata.csv', np.array(Score))

# После всех эпизодов
np.savetxt('Stata.csv', np.array(Score))
plt.plot(Score)
plt.grid()
plt.show()