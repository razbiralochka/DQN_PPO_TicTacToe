import numpy as np

from AZAgent import AZAgent
from CrazyAgent import CrazyAgent
from DQNAgent import DQNAgent
from Enviroment import TicTacToeEnv
from PPOAgent import PPOAgent
import matplotlib.pyplot as plt

env = TicTacToeEnv()

dqnA = DQNAgent()

ppoA = PPOAgent()

alphaZero = AZAgent()

crazyAgent = CrazyAgent()

S = 0
Games = 0
Score = list()

for episode in range(1500):
    env.reset()
    while env.checkBoard() == 3:
        state1 = env.getState()
        #actDqn = dqnA.act(state1)
        actPpo = ppoA.act(state1)

        #actCrazy = crazyAgent.act(state1)

        actPpo, stat = env.step(actPpo,1)

        alphaZero.watchEnemy(state1, actPpo)

        if stat == 3:
            state2 = env.getState()
            #actPpo = ppoA.act(state2)
            #actDqn = dqnA.act(state2)
            actPpoAZ = alphaZero.act(state2, 100)
            #actCrazy = crazyAgent.act(state2)
            #if episode < 1000:
                #actPpoAZ = np.random.randint(0,8)

            actPpoAZ, stat = env.step(actPpoAZ, 2)
        else:
            state2 = env.getState()
            #actPpo = ppoA.act(state2)
            #actDqn = dqnA.act(state2)
            #actPpoAZ = alphaZero.act(state2, 100)
            #actCrazy = crazyAgent.act(state2)

        reward = 0
        if env.checkBoard() == 1:
            reward = 1
        if env.checkBoard() == 2:
            reward = -1


        #dqnA.remember(state1, actDqn, reward, env.getState(), env.checkBoard() != 3)
        ppoA.remember(state2, actPpo, reward)
        #if S < 0:
        #dqnA.replay()

    reward = 0
    if env.checkBoard() == 1:
        reward = 1
    if env.checkBoard() == 2:
        reward = -1

    alphaZero.trainValue(env.getState(), -reward)
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

    #np.savetxt('Stata.csv', np.array(Score))

Score = np.array(Score)
np.savetxt('Stata.csv', Score)

plt.plot(Score)
#plt.ylim([-1.0,1.0])
plt.grid()
plt.show()