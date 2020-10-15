import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np
from entity import entity
from math import sqrt
import copy
import random

def main():
    random.seed(time.time)

    env = gym.make("MsPacman-v0",frameskip=2)

    #Give time for the render to open up
    env.render()
    time.sleep(2)

    numIterations = 100
    scores = list()
    for i in range(numIterations):
        score=0
        env.reset()
        observation,reward,done,_ = env.step(0)
        while not done:
            # Get new state, reward, and if we are done
            action = env.action_space.sample()
            observation,reward,done,_ = env.step(action)
            score += reward
            #env.render()

        print(score)
        scores.append(score)
        #input("Press enter to close window")
    
    print(np.average(scores))

# Only run code if main called this file
if __name__ == "__main__":
    main()