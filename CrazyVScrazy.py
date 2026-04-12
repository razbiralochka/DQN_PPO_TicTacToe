import numpy as np
from Environment import TicTacToeEnv
from CrazyAgent import CrazyAgent
import matplotlib.pyplot as plt

env = TicTacToeEnv()
crazy = CrazyAgent()

S = 0
Games = 0
Score = []

for episode in range(1000):
    env.reset()
    game_over = False

    while not game_over:

        state1 = env.getState()
        actX = crazy.act(state1)
        actX, _ = env.step(actX, 1)  # игнорируем stat, проверим через checkBoard

        result = env.checkBoard()
        if result != 3:
            game_over = True


        if not game_over:
            state2 = env.getState()
            act0 = crazy.act(state2)
            act0, _ = env.step(act0, 2)

            result = env.checkBoard()
            if result != 3:
                game_over = True


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