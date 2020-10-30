import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
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
maxTrainingStep = 300
trainingScoreThreshold = 100

ABSOLUTE_FILE_PATH='~/home/anthony/tensorflowEnv/ai_project'

def getTrainingData(trainingDataSize, maxTrainingSteps, trainingScoreMin):
    trainingData = list()
    kept=0
    for __ in range(trainingDataSize):
        env.reset()

        gameFrames = []
        lastFrame = []
        gameScore = 0

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
                trainingData.append([frame[0][0:180],frame[1]])
            #     possibleFrames = frame[0]
            #     targetData = [possibleFrames,1]
            #     trainingData.append([frame[0],targetData])


    print("Collected training data")
    print("{0} games were samples and {1} were kept".format(trainingDataSize,kept))
    #np.save("TRAINING_DATA_200",np.array(trainingData))
    return trainingData

def seperateTrainingData(trainingDataSet):
    trainingData = []
    targetData = []
    for data in trainingDataSet:
        trainingData.append(data[0])
        targetData.append(data[1])

    return (trainingData,targetData)

def loadInTrainingData(fileName):
    if(os.path.isfile(fileName)):
        trainingData = []
        with np.load(fileName) as data:
            a = data['a']
    else:
        print("not found")
        

def createNetwork(inputDataSize, numValidMoves,learningRate, decayRate):
    # http://tflearn.org/layers/core/
    network = keras.Sequential(
        [
            layers.Conv2D(32,3,3, activation='relu',input_shape=inputDataSize),
            layers.Conv2D(64,3,3,activation='relu'),
            #layers.Dropout(0.8, noise_shape=(batchSize,1,)),
            #layers.Conv2D(32,3,3,activation='relu'),
            layers.Dense(numValidMoves,activation='softmax'),
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

def modelPlay(model, renderGame=False):
    done = False
    scores = []
    for i in range(10):
        env.reset()

        score = 0
        while not done:
            if renderGame:
                env.render()

            #Get new state, reward, and if we are done
            action = getAction(model.predict())
            observation,reward,done,_ = env.step(action)
            
            score += reward

        scores.append(scores)
    
    scores.append(np.average(scores))
    return scores

def main():
    # Get training data

    print("Getting training data")
    trainingDataSet = getTrainingData(numTrainingData,maxTrainingStep,trainingScoreThreshold)
    #trainingDataSet = loadInTrainingData('/trainingData/TRAINING_DATA.npy')

    # Spit training data into raw data and targets
    data, targets = seperateTrainingData(trainingDataSet)

    npData = np.array(data)
    npTarget = np.array(targets)

    # Create model
    print("Creating model")
    # model = createNetwork(inputDataSize=(180,160,3), numValidMoves=possibleMovesLen,learningRate=0.001,decayRate=(0.001/2))
    # model.save('oldModels/uncompiled')

    model = keras.models.load_model('oldModels/uncompiled')

    model.summary()
    

    # Train model
    print("Training model")
    history = model.fit(npData,npTarget,batch_size=64,epochs=2)

    # Save the model for later
    model.save('oldModels/')

    # Run the model on a live game
    testingScores = modelPlay(model)
    print("Average score for 10 games: {0}".format(testingScores[-1]))
    print(testingScores[:-1])

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()