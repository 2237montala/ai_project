import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from gym.envs.classic_control import rendering

def trainModel(network,traingData):
    # Need to take the training data and turn it into inputs and targerts
    pass

def createNetwork(inputDataSize, numValidMoves,lr):
    # http://tflearn.org/layers/core/
    network = input_data(shape=[None,inputDataSize,1], name="input")

    # Create hidden layers
    # This is a complete guess
    network = fully_connected(network, 128, activation='relu')
    # Maybe add dropout after each layer

    network = fully_connected(network, 256, activation='relu')

    network = fully_connected(network, 128, activation='relu')

    # Create output layer
    network = fully_connected(network,numValidMoves,activation='softmax')

    # Create the full network
    network = regression(network,optimizer='sgd',learning_rate=lr)
    
    # Create a object used to train the network
    # Verbrose value increases output but slows down training
    return tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='trainingLogs')

def getAction(predictions):

    pass


def modelPlay(model, renderGame=False):

    env = gym.make("MsPacman-v0")
    done = False

    score = 0
    while not done:
        if renderGame:
            env.render()

        # Get new state, reward, and if we are done
        action = getAction(model.predict())
        observation,reward,done,_ = env.step(action)
        
        score += reward

    return reward

def main():

    # Get training data
    trainingData = list()

    # Train the model on the training data
    model = createNetwork(len(trainingData), 4, 0.001)

    trainMode()

    # Save the model for later

    # Run the model on a live game
    modelPlay(model)

    pass


# Only run code if main called this file
if __name__ == "__main__":
    main()