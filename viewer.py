import numpy as np
import matplotlib.pyplot as plt

Data = np.genfromtxt('Stata.csv', delimiter=';')

#for i in range(len(Data)):
    #Data[i] = Data[i]/(i+1)

plt.plot(Data)
plt.grid()
#plt.xlim([0,500])
#plt.ylim([-50,50])
plt.ylabel('Победы(Х) - Победы(0)')
plt.xlabel('Партии')
plt.show()