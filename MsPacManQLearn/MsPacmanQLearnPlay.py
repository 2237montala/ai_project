"""
    This program runs the saved model on a set of games
    Models can be stored any where but by default it looks in the 
    folder 300EpochTrain

    The number of games can be specified
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from gym.envs.classic_control import rendering
import random as random
from FrameStack import FrameStack
from DQNModel import DQNModel
import time

FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160
FRAME_REDUCTION = 2
FRAME_STACKING = 4
INPUT_FRAME_SIZE = (int(FRAME_X_SIZE/FRAME_REDUCTION),int(FRAME_Y_SIZE/FRAME_REDUCTION),FRAME_STACKING)

def preprocessFrame(frameIn, sizeX):
    # Make array into numpy array
    # Cut off bottom of image
    frameNp = np.array(frameIn[0:sizeX])

    # Grey scale image
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    grayFrame = np.dot(frameNp[...,:3],[0.2989,0.5870,0.1140])

    # Reduce the size of the image by 2
    # https://medium.com/gradientcrescent/fundamentals-of-reinforcement-learning-automating-pong-in-using-a-policy-model-an-implementation-b71f64c158ff
    # In the section about preparing image
    grayFrame = grayFrame[::FRAME_REDUCTION,::FRAME_REDUCTION]

    # Return and turn values to 1 byte values to save space
    return grayFrame.astype(np.uint8)


targetModel = keras.models.load_model('./MsPacManQLearn/oldModels/300EpochTrain/target')

# Change this number to change how many games a played
gamesToPlay = 10
env = gym.make("MsPacman-v0")

# If true then the game is rendered on screen
# If false nothing is shown on screen
renderGame = True

scores = []
for _ in range(gamesToPlay):
    # Create frame stacking
    stacked_frames = FrameStack(FRAME_STACKING)
    stacked_state = stacked_frames.reset(preprocessFrame(env.reset(),FRAME_X_SIZE))

    done = False
    score = 0
    frame_count = 0
    action = 0
    while not done:
        frame_count +=1
        
        #Get new state, reward, and if we are done
        if frame_count % FRAME_STACKING == 0:
            temp = stacked_state.reshape((1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))
            action = np.argmax(targetModel.predict(temp))

        state_single,reward,done,_ = env.step(action)
        stacked_state = stacked_frames.step(preprocessFrame(state_single,FRAME_X_SIZE))

        if renderGame:
            env.render()
            time.sleep(0.033)

        score += reward

    scores.append(score)

print("Average score for {0} games: {1}".format(10,np.average(gamesToPlay)))