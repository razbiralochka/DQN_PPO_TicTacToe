import numpy as np

from DQNAgent import DQNAgent
from Enviroment import TicTacToeEnv
from PPOAgent import PPOAgent
import matplotlib.pyplot as plt

env = TicTacToeEnv()

dqnA = DQNAgent()

ppoA = PPOAgent()

S = 0
Games = 0
Score = list()

for episode in range(2000):
    env.reset()
    while env.checkBoard() == 3:
        state = env.getState()
        actDqn = dqnA.act(state)
        act, stat = env.step(actDqn,1)

        reward = 0
        if stat == 1:
            reward = 1
        if stat == 2:
            reward = -1

        dqnA.remember(state, act, reward, env.getState(), env.checkBoard() != 3)


        if(env.checkBoard() == 3):
            state = env.getState()
            actPpo = ppoA.act(state)
            act, stat = env.step(actPpo,2)


            reward = 0
            if stat == 2:
                reward = 1
            if stat == 1:
                reward = -1
            ppoA.remember(state, act, reward)

        else:
            state = env.getState()
            actPpo = ppoA.act(state)
            stat = env.checkBoard()
            reward = 0
            if stat == 2:
                reward = 1
            if stat == 1:
                reward = -1
            ppoA.remember(state, actPpo, reward)
            break

    stat = env.checkBoard()
    Games += 1
    if stat == 2:
        S += 1
        Score.append(S)
        print(Games, env.board, ' ZeroWin ', S)
    if stat == 1:
        S -= 1
        Score.append(S)
        print(Games, env.board,' CrossWin ', S)
    if stat == 0:
        Score.append(S)
        print(Games, env.board, ' Nothing ', S)
    ppoA.rememberTraj()
    dqnA.replay()
    ppoA.learn()

Score = np.array(Score)

np.savetxt('Stata', Score)

plt.plot(Score)
#plt.ylim([-1.0,1.0])
plt.grid()
plt.show()