import random

class CrazyAgent:
    def act(self,state):
        available = [i for i, x in enumerate(state) if x == 0]
        return random.choice(available)