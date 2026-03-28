import numpy as np
from PIL.ImagePalette import random
from math import floor

from numpy.random import randint

from AZAgent import AZAgent
from CrazyAgent import CrazyAgent
from DQNAgent import DQNAgent
from Enviroment import TicTacToeEnv
from PPOAgent import PPOAgent
import matplotlib.pyplot as plt
import random


env = TicTacToeEnv()


alphaZero = AZAgent()
dqnA = DQNAgent()

sims = 0
S = 0
Games = 0
Score = list()

lastA0 = 0
lastS0 = 0

for episode in range(1000):
    env.reset()
    reward = 0
    while env.checkBoard() == 3:
        state1 = env.getState()

        actX = dqnA.act(state1)

        if episode < 200:
            actx = random.choice([i for i, cell in enumerate(state1) if cell == 0])

        actX, stat = env.step(actX,1)

        state2 = env.getState()
        act0 = 0
        if stat == 3:
            sims = 1 + 10 * floor(episode / 100)
            act0 = alphaZero.act(state2, 100, 2)
            act0, stat = env.step(act0, 2)
        else:
            act0 = 0

        if env.checkBoard() == 1:
            reward = 1
        if env.checkBoard() == 2:
            reward = -1

        dqnA.remember(state1, actX, reward, env.getState(), env.checkBoard() != 3)
        dqnA.replay()

    alphaZero.trainValue()


    stat = env.checkBoard()
    Games += 1
    if stat == 2:
        S -= 1
        sims = sims -1
        Score.append(S)
        print(Games, env.board, ' ZeroWin ', S)
    if stat == 1:
        S += 1
        sims = sims + 1
        sims = min(sims, 100)
        Score.append(S)
        print(Games, env.board,' CrossWin ', S, sims)
    if stat == 0:
        Score.append(S)
        print(Games, env.board, ' Nothing ', S)

    if episode % 100 ==0:
        np.savetxt('Stata.csv', np.array(Score))



Score = np.array(Score)
np.savetxt('Stata.csv', Score)

plt.plot(Score)
#plt.ylim([-1.0,1.0])
plt.grid()
plt.show()