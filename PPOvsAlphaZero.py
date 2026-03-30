import numpy as np
from AZAgent import AZAgent
from PPOAgent import PPOAgent
from Enviroment import TicTacToeEnv
import matplotlib.pyplot as plt
from CrazyAgent import CrazyAgent


env = TicTacToeEnv()

alphaZero = AZAgent()
#ppoA = PPOAgent()
a = CrazyAgent()

S = 0
Games = 0
Score = list()


for episode in range(1000):
    env.reset()
    reward = 0
    while env.checkBoard() == 3:
        state1 = env.getState()

        actX = a.act(state1)

        actX, stat = env.step(actX,1)

        state2 = env.getState()
        #ppoA.remember(state1, actX, reward)
        if env.checkBoard() != 3:
            break
        act0 = alphaZero.act(state2, sims=200)
        act0, stat = env.step(act0, 2)


        if env.checkBoard() == 1:
            reward = 1
        if env.checkBoard() == 2:
            reward = -1


    _ ,v = alphaZero.get_policy_value(env.getState(),2)
    #ppoA.rememberTraj()
    #ppoA.learn()
    stat = env.checkBoard()
    Games += 1
    if stat == 2:
        S -= 1
        Score.append(S)
        print(Games, env.board, 'ZeroWin ', S, v)
    if stat == 1:
        S += 1
        Score.append(S)
        print(Games, env.board,'CrossWin ', S, v)
    if stat == 0:
        Score.append(S)
        print(Games, env.board, 'Nothing ', S)

    if episode % 100 ==0:
        np.savetxt('Stata.csv', np.array(Score))



Score = np.array(Score)
np.savetxt('Stata.csv', Score)

plt.plot(Score)
plt.grid()
plt.show()