import numpy as np
from DQNAgent import DQNAgent
from Environment import TicTacToeEnv
from AZAgent import AZAgent
import matplotlib.pyplot as plt

env = TicTacToeEnv()
dqnA = DQNAgent()
alphaZero = AZAgent()

S = 0
Games = 0
Score = []

for episode in range(1000):
    env.reset()
    game_over = False

    while not game_over:
        # Ход DQN (крестики, 1)
        state1 = env.getState()
        actX = dqnA.act(state1)
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
            az_reward = -dqn_reward
        else:
            dqn_reward = 0
            az_reward = 0

        # Запоминаем переход DQN: (s, a, r, s', done)
        next_state = env.getState()


        # Если игра не окончена — ход PPO (нолики, 2)
        if not game_over:
            state2 = env.getState()
            act0 = alphaZero.act(state2, sims=200)
            act0, _ = env.step(act0, 2)

            result = env.checkBoard()
            if result != 3:
                game_over = True
                if result == 2:
                    az_reward = 1
                elif result == 1:
                    az_reward = -1
                else:
                    az_reward = 0
                dqn_reward = -az_reward  # на случай, если DQN ещё будет учиться (но уже не будет)
            else:
                az_reward = 0

            dqnA.remember(state1, actX, dqn_reward, env.getState(), game_over)
            dqnA.replay()  # обучаем DQN

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

    # Сохранение каждые 100 эпизодов
    if episode % 100 == 0:
        np.savetxt('Stata.csv', np.array(Score))

# Финал
Score = np.array(Score)
np.savetxt('Stata.csv', Score)
plt.plot(Score)
plt.grid()
plt.show()