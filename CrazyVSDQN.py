import numpy as np
from DQNAgent import DQNAgent
from Enviroment import TicTacToeEnv
from CrazyAgent import CrazyAgent
import matplotlib.pyplot as plt
import random


env = TicTacToeEnv()


dqnA = DQNAgent()
crazy = CrazyAgent()


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
        actX = crazy.act(state1)
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
            dqn_reward = -ppo_reward
        else:
            ppo_reward = 0
            dqn_reward = 0  # игра продолжается

        # Если DQN делал ход ранее — запоминаем переход (s, a, r, s', done)
        if state2 is not None and act0 is not None:
            next_state = env.getState()
            dqnA.remember(state2, act0, dqn_reward, next_state, game_over)

        # Если игра не окончена — DQN делает ход
        if not game_over:
            state2 = env.getState()
            act0 = dqnA.act(state2)
            act0, stat = env.step(act0, 2)

            # Проверяем, закончилась ли игра после хода DQN
            result = env.checkBoard()
            if result != 3:
                game_over = True
                if result == 2:
                    dqn_reward = 1
                elif result == 1:
                    dqn_reward = -1
                else:
                    dqn_reward = 0
                ppo_reward = -dqn_reward
            else:
                dqn_reward = 0
                ppo_reward = 0


        # Если игра окончена, запоминаем последний переход DQN (если он был)
        if game_over and state2 is not None and act0 is not None:
            final_state = env.getState()
            dqnA.remember(state2, act0, dqn_reward, final_state, True)

        # Обучение
        dqnA.replay()


    # Логирование и подсчёт счёта
    result = env.checkBoard()
    Games += 1
    if result == 2:
        S -= 1
        print(Games, env.board, ' ZeroWin ', S)
    elif result == 1:
        S += 1
        print(Games, env.board, ' CrossWin ', S)
    else:
        print(Games, env.board, ' Draw ', S)
    Score.append(S)

    if episode % 100 == 0:
        np.savetxt('Stata.csv', np.array(Score))

# После всех эпизодов
np.savetxt('Stata.csv', np.array(Score))
plt.plot(Score)
plt.grid()
plt.show()