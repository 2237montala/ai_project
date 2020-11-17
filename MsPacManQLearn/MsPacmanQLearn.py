import gym
import numpy as np
import tensorflow as tf
from tf import keras
from tf.keras import layers
import os
import sys
import datetime
#from keras.callbacks import TensorBoard
from gym.envs.classic_control import rendering
from tf.keras.callbacks import ModelCheckpoint
from OldAI import OldAi
import matplotlib.pyplot as plt

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
FRAME_REDUCTION = 2
INPUT_FRAME_SIZE = (FRAME_X_SIZE/FRAME_REDUCTION,FRAME_Y_SIZE/FRAME_REDUCTION,1)

NUM_EPOCHS = 800 # Changes how many times we run q learning
BATCH_SIZE = 2   # Number of games per training session
LEARNING_RATE = 0.001 
DISCOUNT_FACTOR = 0.97 # How much the current state reward is reduced by

epsilon = 0.5   # Current epsilon values
eps_min = 0.05  
eps_max = 1.0
eps_decay_steps = 50000 # How often epsilon gets recalcuated


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

    # Reduce the size of the image by 2
    # https://medium.com/gradientcrescent/fundamentals-of-reinforcement-learning-automating-pong-in-using-a-policy-model-an-implementation-b71f64c158ff
    # In the section about preparing image
    grayFrame = grayFrame[::FRAME_REDUCTION,::FRAME_REDUCTION]

    # Return and turn values to 1 byte values to save space
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
            # Might run out of vram
            layers.Input(shape=inputDataSize),
            layers.Conv2D(32,(8,8),4, activation='relu'),
            layers.Conv2D(64,(4,4),2, activation='relu'),
            layers.Conv2D(64,(3,3),1, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(env.action_space.n,activation='softmax'),

            #layers.Dropout(0.5),
            #layers.MaxPool2D(pool_size=(2,2)),
            #layers.Dense(16, activation='relu'),
        ]
    )

    #optimizer = keras.optimizers.Adam(learning_rate=learningRate, clipnorm=1.0)
    #sgd = tf.keras.optimizers.SGD(lr=learningRate, decay=decayRate, momentum=0.8)

    # network.compile(loss='categorical_crossentropy',
    #                 optimizer=optimizer,
    #                 metrics=['accuracy'])

    #network.compile(metrics=['accuracy'])

    return network

def epsilonGreedy(networkModel, state, step):
    # Learn about it here
    # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
    # Calcualte new epsilion value
    # Something like this???
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps) #Decaying policy with more steps

    if np.random.rand() < epsilon:
        return exploreAction()
    else:
        return exploitAction(networkModel,state)

def exploreAction():
    return env.action_space.sample()

def exploitAction(networkModel, state):
    return networkModel.predict(state)

def getAction(predictions):
    # From the prediction get the best move
    predictionIndex = np.argmax(predictions)

    return possibleMoves[predictionIndex]

def getBatchGameData(prev_states,prev_actions,next_states,reward_history,done_history, batchSize):
    # Idea came from here
    # https://towardsdatascience.com/automating-pac-man-with-deep-q-learning-an-implementation-in-tensorflow-ca08e9891d9c
    # Where they talk about sample memory
    # Generate a list of indices using the batch number
    batch_indices = np.random.choice(range(len(prev_states)), size=batchSize)

    # Create a random selection of indicies from our batch indicies
    batch_prev_state = np.array([prev_states[i] for i in batch_indices])
    batch_prev_state_action = np.array([prev_actions[i] for i in batch_indices])
    batch_next_states = np.array([next_states[i] for i in batch_indices])
    batch_reward = np.array([reward_history[i] for i in batch_indices])
    batch_done_flags = np.array(done_history[i] for i in batch_indices)

    return batch_prev_state,batch_prev_state_action,batch_next_states,batch_reward,batch_done_flags

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

def qLearn(trainingModel, targetModel):
    # Set up the optimizer function
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # Set up loss function
    loss_func = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

    # Set up q learning historical buffers
    prev_states = []
    prev_state_action = []
    next_states = []
    reward_history = []
    done_flags = []
    num_states_in_history = 1000
    train_model_after_num_actions = 4
    update_target_model_after_num_epochs = 1000
    frame_count = 0 # Number of frames seen
    epochs_ran = 0

    trained = False
    while not trained:
        state = env.reset()
        epoch_reward = 0

        for i in range(1,num_states_in_history):
            frame_count += 1

            # Get action based on epsilion greedy algorithm
            # As we see more frames we will explore less and exploit more
            action = epsilonGreedy(trainingModel,state,frame_count)

            # Do the action
            state_next,reward,done,_ = env.step(action)

            # Update the historical lists
            prev_states.append(state)
            prev_state_action.append(action)
            next_states.append(state_next)
            reward_history.append(reward)
            done_flags.append(done)
            epoch_reward += reward

            # Update the current state
            state = state_next          

            # Train the training model if we moved enough times
            if i % train_model_after_num_actions:
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = getBatchGameData(prev_states,prev_state_action,next_states,reward_history,done_flags,BATCH_SIZE)

                # Generate all the rewards for the next states
                possible_rewards = targetModel.predict(batch_next_states)  

                # Generate q values from all the next states
                # Need to use tf.reduce_max because the possible_rewards is a tensor not an numpy array
                q_values = batch_rewards + (DISCOUNT_FACTOR * tf.reduce_max(possible_rewards, axis=1))

                # We need to penalize a death
                # Check each index in batch_done for if the game is over
                for j in range(BATCH_SIZE):
                    if batch_done[j] == True:
                        q_values[j] = -1

                # Create a mask so that we can ignore q values that didn't change
                # Makes a one hot vector where each action is represented in binary
                # https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
                action_mask = tf.one_hot(batch_actions, possibleMovesLen)

                # GradientTape monitors what calculations are done inside the
                # statement so we can backpropigate using tensorflow functions
                with tf.GradientTape() as tape:
                    # Now we need to update the training model
                    # We should find the difference in q values between the training model
                    # and the target model. Incase our training model got worse
                    orig_q_value = trainingModel(batch_states)

                    # Generate the best action based on our q values. Using the one hot
                    # mask we elimiate action that are equal to 0 and include actions
                    # that are equal to 1. q_action is a number we want to maximise
                    q_action = tf.reduce_sum(tf.multiply(orig_q_value,action_mask),axis=1)

                    # Calcuate the loss between the q_action and all the q values
                    loss = loss_func(q_values,q_action)

                # Using tape, calculate the gradients and back propigate
                gradients = tape.gradient(loss, trainingModel.trainable_variables)
                optimizer.apply_gradients(zip(gradients,trainingModel.trainable_variables))

            # If we have done enought epochs then we should update the target model
            if i % update_target_model_after_num_epochs == 0:
                targetModel.set_weights(trainingModel.get_weights())

            # Clear out ram for the buffer if we reach max length
            if len(done_flags) > num_states_in_history:
                del prev_states[:1]
                del prev_state_action[:1]
                del next_states[:1]
                del reward_history[:1]
                del done_flags[:1]

        # Need to come up with break condition
        if epochs_ran == NUM_EPOCHS:
            trained = True
        else:
            epochs_ran += 1


def main():
    
    # Create training model
    train_model = createNetwork(inputDataSize=INPUT_FRAME_SIZE, numValidMoves=possibleMovesLen,learningRate=0.1,decayRate=0)

    # Create target model
    target_model = createNetwork(inputDataSize=INPUT_FRAME_SIZE, numValidMoves=possibleMovesLen,learningRate=0.1,decayRate=0)

    qLearn(trainingModel=train_model,targetModel=target_model)
    
    # # Set up periodic saving of models
    # # checkpoint = ModelCheckpoint("bestModel/", monitor='accuracy', verbose=1,
    # # save_best_only=True, mode='max')


    # #model = keras.models.load_model('oldModels/try4/')

    # #model.summary()

    # # batchSize = 2
    # # #dataGenTrain = DataGenerator(batchSize=batchSize, gamesToRun=numGamesToTrainOn,maxTrainingSteps=maxTrainingStep,trainingScoreMin=trainingScoreThreshold)
    # # dataGenTrain = DataGenerator(batchSize=batchSize, gamesToRun=numGamesToTrainOn,maxTrainingSteps=maxTrainingStep,trainingScoreMin=600)

    # # #trainGen = dataGenTrain.getTrainingData()

    # # model.fit(dataGenTrain,steps_per_epoch=numGamesToTrainOn / batchSize,
    # #             epochs=7, callbacks=[checkpoint], use_multiprocessing=True, workers=2)

    # # model.save('oldModels/try6')
    
    # #Run the model on a live game
    # print("Testing models")
    # testingScores, average = modelPlay(model,1,renderGame=True)
    
    # print("Average score for {0} games: {1}".format(100,average))
    # print(testingScores)

    # # input("Press enter to watch a live game")
    # # testingScores = modelPlay(model,renderGame=True)

    # # with open("scores.csv",'w') as f:
    # #     for score in testingScores:
    # #         f.write("{0},\n".format(score))

    # #     f.write("{0},\n".format(average))
    

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()