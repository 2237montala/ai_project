import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import datetime
#from keras.callbacks import TensorBoard
from gym.envs.classic_control import rendering
from tensorflow.keras.callbacks import ModelCheckpoint
from OldAI import OldAi

# Create game
env = gym.make("MsPacman-v0",frameskip=2)

#                UP DOWN LEFT RIGHT
possibleMoves = [2   ,5  ,3   ,4]
possibleMovesLen = len(possibleMoves)
numGamesToTrainOn = 1000
maxTrainingStep = 1000
trainingScoreThreshold = 250

FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,batchSize, gamesToRun, maxTrainingSteps, trainingScoreMin):
        self.batch_size = batchSize
        self.gameSteps = maxTrainingStep
        self.gamesToRun = gamesToRun
        self.gameMinScore = trainingScoreMin
        self.trainAiType = OldAi((180,160))
    def __len__(self):
        return self.gamesToRun // self.batch_size

    def __getitem__(self,index):
        #return self.getTrainingData()
        return self.getTrainingDataOldAi()

    def getTrainingDataOldAi(self):
        trainingData = list()
        kept=0
        while(kept < self.batch_size):
            env.reset()

            gameFrames = []
            gameScore = 0

            gameScore, gameFrames = self.trainAiType.runGame(env,self.gameSteps)

            if gameScore > self.gameMinScore:
                kept +=1
                # Good enough game so add it to training data
                for frame in gameFrames:
                    # Make a target array the length of the possible actions

                    temp = np.zeros(env.action_space.n) 

                    # Fill in a 1 where this frame did its action
                    temp[frame[1]] = 1

                    trainingData.append([preprocessFrame(frame[0],FRAME_X_SIZE),temp])

        return seperateTrainingData(trainingData)

    def getTrainingData(self):
        trainingData = list()
        kept=0
        while(kept < self.batch_size):
            env.reset()

            gameFrames = []
            lastFrame = []
            gameScore = 0

            for _ in range(80):
                action = env.action_space.sample()
                lastFrame, reward, done, info = env.step(action)

            # Run a test game
            for i in range(self.gameSteps):
                action = env.action_space.sample()

                if i > 0:
                    gameFrames.append([lastFrame,action])

                lastFrame, reward, done, info = env.step(action)
                gameScore += reward
                if done:
                    break

            if gameScore > self.gameMinScore:
                kept +=1
                # Good enough game so add it to training data
                for frame in gameFrames:
                    # Make a target array the length of the possible actions

                    temp = np.zeros(env.action_space.n) 

                    # Fill in a 1 where this frame did its action
                    temp[frame[1]] = 1

                    trainingData.append([preprocessFrame(frame[0],FRAME_X_SIZE),temp])

        return seperateTrainingData(trainingData)

def preprocessFrame(frameIn, sizeX):
    # Make array into numpy array
    # Cut off bottom of image
    frameNp = np.array(frameIn[0:sizeX])

    # Grey scale image
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    grayFrame = np.dot(frameNp[...,:3],[0.2989,0.5870,0.1140])
    return grayFrame.astype(np.uint8)

def seperateTrainingData(trainingDataSet):
    trainingData = []
    targetData = []
    for data in trainingDataSet:
        trainingData.append(data[0])
        targetData.append(data[1])

    trainingDataNp = np.array(trainingData)
    trainingDataNp = trainingDataNp.reshape((trainingDataNp.shape[0],
                                            trainingDataNp.shape[1],
                                            trainingDataNp.shape[2],1))
    return (trainingDataNp,np.array(targetData,dtype=np.uint8))        

def createNetwork(inputDataSize, numValidMoves,learningRate, decayRate):
    network = keras.Sequential(
        [
            layers.Input(shape=inputDataSize),
            #layers.Conv2D(16,2,2, activation='relu'),
            layers.Conv2D(16,(2,2),2, activation='relu'),
            #layers.Conv2D(16,3,2, activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Dropout(0.5),
            layers.Conv2D(16,3,2, activation='relu'),
            layers.Conv2D(8,3,2, activation='relu'),
            #layers.MaxPool2D(pool_size=(2,2)),
            layers.Dropout(0.5),
            #layers.Dense(16, activation='relu'),
            layers.Flatten(),
            layers.Dense(env.action_space.n,activation='softmax'),
        ]
    )

    sgd = tf.keras.optimizers.SGD(lr=learningRate, decay=decayRate, momentum=0.8)

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
        #print("Game {0}".format(i+1))
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
            npDoubleObv = npDoubleObv.reshape((npDoubleObv.shape[0],
                                            npDoubleObv.shape[1],
                                            npDoubleObv.shape[2],1))

            action = np.argmax(model.predict(npDoubleObv))
            #print(action)
            observation,reward,done,_ = env.step(action)
            
            score += reward

        scores.append(score)
    return scores,np.average(scores)

def main():
    # # Create model
    # print("Creating model")
    # model = createNetwork(inputDataSize=(FRAME_X_SIZE,FRAME_Y_SIZE,1), numValidMoves=possibleMovesLen,learningRate=0.1,decayRate=(0.001/2))

    # Set up periodic saving of models
    # checkpoint = ModelCheckpoint("bestModel/", monitor='accuracy', verbose=1,
    # save_best_only=True, mode='max')


    model = keras.models.load_model('oldModels/try4/')

    model.summary()

    # batchSize = 2
    # #dataGenTrain = DataGenerator(batchSize=batchSize, gamesToRun=numGamesToTrainOn,maxTrainingSteps=maxTrainingStep,trainingScoreMin=trainingScoreThreshold)
    # dataGenTrain = DataGenerator(batchSize=batchSize, gamesToRun=numGamesToTrainOn,maxTrainingSteps=maxTrainingStep,trainingScoreMin=600)

    # #trainGen = dataGenTrain.getTrainingData()

    # model.fit(dataGenTrain,steps_per_epoch=numGamesToTrainOn / batchSize,
    #             epochs=7, callbacks=[checkpoint], use_multiprocessing=True, workers=2)

    # model.save('oldModels/try6')
    
    #Run the model on a live game
    print("Testing models")
    testingScores, average = modelPlay(model,1,renderGame=True)
    
    print("Average score for {0} games: {1}".format(100,average))
    print(testingScores)

    # input("Press enter to watch a live game")
    # testingScores = modelPlay(model,renderGame=True)

    # with open("scores.csv",'w') as f:
    #     for score in testingScores:
    #         f.write("{0},\n".format(score))

    #     f.write("{0},\n".format(average))
    

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()