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

for episode in range(10000):
    env.reset()
    while env.checkBoard() == 3:
        state1 = env.getState()
        actDqn = dqnA.act(state1)
        #actPpo = ppoA.act(state1)

        act, stat = env.step(actDqn,1)

        if stat == 3:
            state2 = env.getState()
            actPpo = ppoA.act(state2)
            #actDqn = dqnA.act(state2)
            act, stat = env.step(actPpo, 2)
        else:
            state2 = env.getState()
            actPpo = ppoA.act(state2)
            #actDqn = dqnA.act(state2)
            act = actPpo

        reward = 0
        if env.checkBoard() == 1:
            reward = 1
        if env.checkBoard() == 2:
            reward = -1

        dqnA.remember(state1, act, reward, env.getState(), env.checkBoard() != 3)
        ppoA.remember(state2, act, -reward)
        #if S < 0:
        dqnA.replay()

    ppoA.rememberTraj()
    #if S > 0:
    ppoA.learn()
    stat = env.checkBoard()
    Games += 1
    if stat == 2:
        S -= 1
        Score.append(S)
        print(Games, env.board, ' ZeroWin ', S)
    if stat == 1:
        S += 1
        Score.append(S)
        print(Games, env.board,' CrossWin ', S)
    if stat == 0:
        Score.append(S)
        print(Games, env.board, ' Nothing ', S)

    np.savetxt('Stata.csv', np.array(Score))

Score = np.array(Score)
np.savetxt('Stata.csv', Score)

plt.plot(Score)
#plt.ylim([-1.0,1.0])
plt.grid()
plt.show()