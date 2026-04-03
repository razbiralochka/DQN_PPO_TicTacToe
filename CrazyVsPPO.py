import numpy as np
from CrazyAgent import CrazyAgent
from Enviroment import TicTacToeEnv
from PPOAgent import PPOAgent
import matplotlib.pyplot as plt

env = TicTacToeEnv()
crazy = CrazyAgent()
ppoA = PPOAgent()

S = 0
Games = 0
Score = []

for episode in range(1000):
    env.reset()
    game_over = False

    while not game_over:
        # Ход DQN (крестики, 1)
        state1 = env.getState()
        actX = crazy.act(state1)
        actX, _ = env.step(actX, 1)  # игнорируем stat, проверим через checkBoard

        result = env.checkBoard()
        if result != 3:
            game_over = True
            # Определяем награду для DQN
            if result == 1:
                dqn_reward = 1
            elif result == 2:
                dqn_reward = -1
            else:  # ничья
                dqn_reward = 0
            ppo_reward = -dqn_reward
        else:
            dqn_reward = 0
            ppo_reward = 0


        # Если игра не окончена — ход PPO (нолики, 2)
        if not game_over:
            state2 = env.getState()
            act0 = ppoA.act(state2)
            act0, _ = env.step(act0, 2)

            result = env.checkBoard()
            if result != 3:
                game_over = True
                if result == 2:
                    ppo_reward = 1
                elif result == 1:
                    ppo_reward = -1
                else:
                    ppo_reward = 0
                dqn_reward = -ppo_reward  # на случай, если DQN ещё будет учиться (но уже не будет)
            else:
                ppo_reward = 0

            # Запоминаем действие PPO
            ppoA.remember(state2, act0, ppo_reward)


    # Конец игры: обучаем PPO
    ppoA.rememberTraj()
    ppoA.learn()

    # Обновляем счёт
    result = env.checkBoard()
    Games += 1
    if result == 1:
        S += 1
        print(Games, env.board, ' CrossWin ', S)
    elif result == 2:
        S -= 1
        print(Games, env.board, ' ZeroWin ', S)
    else:
        print(Games, env.board, ' Draw ', S)
    Score.append(S)

    # Сохранение каждые 100 эпизодов
    if episode % 100 == 0:
        np.savetxt('Stata.csv', np.array(Score))

# Финал
Score = np.array(Score)
np.savetxt('Stata.csv', Score)
plt.plot(Score)
plt.grid()
plt.show()