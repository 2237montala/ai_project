import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
from gym.envs.classic_control import rendering

# Create game
env = gym.make("MsPacman-v0")

#                UP DOWN LEFT RIGHT
possibleMoves = [2   ,5  ,3   ,4]
possibleMovesLen = len(possibleMoves)
numTrainingData = 50
maxTrainingStep = 700
trainingScoreThreshold = 300

FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160

ABSOLUTE_FILE_PATH='~/home/anthony/tensorflowEnv/ai_project'

def getTrainingData(trainingDataSize, maxTrainingSteps, trainingScoreMin):
    trainingData = list()
    kept=0
    while(kept < trainingDataSize):
        env.reset()

        gameFrames = []
        lastFrame = []
        gameScore = 0

        for _ in range(80):
            action = env.action_space.sample()
            lastFrame, reward, done, info = env.step(action)

        # Run a test game
        for i in range(maxTrainingSteps):
            # Enable if you want to see training data
            #env.render()
            action = env.action_space.sample()

            if i > 0:
                gameFrames.append([lastFrame,action])

            lastFrame, reward, done, info = env.step(action)
            gameScore += reward
            if done:
                break

        #print(gameScore)
        if gameScore > trainingScoreMin:
            kept +=1
            # Good enough game so add it to training data
            for frame in gameFrames:
                # Make a target array the length of the possible actions
                temp = np.zeros(env.action_space.n) 

                # Fill in a 1 where this frame did its action
                temp[frame[1]] = 1


                trainingData.append([preprocessFrame(frame[0],FRAME_X_SIZE),temp])

            if(kept % int(trainingDataSize/10) == 0):
                print("{0} games out of {1} saved".format(kept,trainingDataSize))


    print("Collected training data")
    #np.save("TRAINING_DATA_300_at_score_250",np.array(trainingData))
    return trainingData

def preprocessFrame(frameIn, sizeX):
    # Make array into numpy array
    # Cut off bottom of image
    frameNp = np.array(frameIn[0:sizeX])

    # Grey scale image
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    np.dot(frameIn[...,:3],[0.2989,0.5870,0.1140])

    # Reshape array to be x,y, (gray scale)
    frameNp = frameNp.reshape(frameNp.shape[1],1)

    return frameNp

def seperateTrainingData(trainingDataSet):
    trainingData = np.array(trainingDataSet[0:][0])
    targetData = np.array(trainingDataSet[0:][1])

    return (trainingData,targetData)

def loadInTrainingData(fileName):
    if(os.path.isfile(fileName)):
        trainingData = []
        with np.load(fileName) as data:
            a = data['a']
    else:
        print("not found")
        

def createNetwork(inputDataSize, numValidMoves,learningRate, decayRate):
    network = keras.Sequential(
        [
            layers.Input(shape=inputDataSize),
            layers.Conv2D(32,3,2, activation='relu'),
            layers.Conv2D(64,3,2, activation='relu'),
            layers.Dropout(0.8),
            layers.Dense(32,activation='relu'),
            layers.Flatten(),
            layers.Dense(env.action_space.n,activation='softmax'),
        ]
    )

    sgd = tf.keras.optimizers.SGD(lr=learningRate, decay=decayRate, momentum=0.5)

    network.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

    return network


def getAction(predictions):
    # From the prediction get the best move
    predictionIndex = np.argmax(predictions)

    return possibleMoves[predictionIndex]

def modelPlay(model, gamesToPlay=1, renderGame=False):
    
    scores = []
    for i in range(gamesToPlay):
        print("Game {0}".format(i))
        observation = env.reset()
        done = False

        score = 0
        while not done:
            if renderGame:
                env.render()

            #Get new state, reward, and if we are done
            procdArr = list()
            procdArr.append(preprocessFrame(observation,FRAME_X_SIZE))
            npDoubleObv = np.array(procdArr)

            action = np.argmax(model.predict(npDoubleObv))
            observation,reward,done,_ = env.step(action)
            
            score += reward

        scores.append(scores)
    
    scores.append(np.average(scores))
    return scores

def main():
    # Create model
    print("Creating model")
    model = createNetwork(inputDataSize=(FRAME_X_SIZE,FRAME_Y_SIZE,1), numValidMoves=possibleMovesLen,learningRate=0.001,decayRate=(0.001/2))
    # #model = keras.models.load_model('oldModels/uncompiled')
    model.summary()

    for i in range(10):
        print("Getting training data")
        trainingDataSet = getTrainingData(numTrainingData,maxTrainingStep,trainingScoreThreshold)

        #trainingDataSet = loadInTrainingData('/trainingData/TRAINING_DATA.npy')

        # Spit training data into raw data and targets
        data, targets = seperateTrainingData(trainingDataSet)

        if(len(data) == 0):
            sys.exit()

        # Train model
        print("Training model")
        history = model.fit(data,targets,batch_size=32,epochs=5)
        

    model.save('oldModels/')
    print("Loading in old model")
    model = keras.models.load_model('oldModels')

    # Run the model on a live game
    print("Testing models")
    testingScores = modelPlay(model,20)
    
    print("Average score for {0} games: {1}".format(20,testingScores[-1]))
    print(testingScores[:-1])

    testingScores = modelPlay(model,renderGame=True)
    input()

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()