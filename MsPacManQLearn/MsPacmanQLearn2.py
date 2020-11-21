import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from gym.envs.classic_control import rendering
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random as random
from FrameStack import FrameStack
from DQNModel import DQNModel
#from gym.wrappers import FrameStack


# More references
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c

# Create game
env = gym.make("MsPacman-v0",frameskip=4)
#env = FrameStack(env,4)

#possibleMoves = [2,3,4,5]
possibleMoves = [0,1,2,3,4,5,6,7,8]
possibleMovesLen = len(possibleMoves)


FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160
FRAME_REDUCTION = 2
FRAME_STACKING = 4
INPUT_FRAME_SIZE = (int(FRAME_X_SIZE/FRAME_REDUCTION),int(FRAME_Y_SIZE/FRAME_REDUCTION),FRAME_STACKING)

NUM_EPOCHS = 20 # Changes how many times we run q learning
NUM_STEPS_PER_EPOCH = 10000 # How many frames will be ran through each epoch
BATCH_SIZE = 32   # Number of games per training session
LEARNING_RATE = 0.001 
DISCOUNT_FACTOR = 0.99 # How much the current state reward is reduced by

epsilon = 1.0 # Current epsilon values
eps_min = 0.1 #0.05
eps_max = 1.0
eps_decay_steps = 100000#1000000.0 # this value specifies how many frame we need to see before
                            # we switch from explore to exploit



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
    batch_done_flags = np.array([done_history[i] for i in batch_indices], dtype=np.uint8)

    return batch_prev_state,batch_prev_state_action,batch_next_states,batch_reward,batch_done_flags

def modelPlay(DQNModel, gamesToPlay=1, renderGame=False):
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
                action = DQNModel.exploitAction(temp)

            state_single,reward,done,_ = env.step(action)
            stacked_state = stacked_frames.step(preprocessFrame(state_single,FRAME_X_SIZE))

            if renderGame:
                env.render()

            score += reward

        scores.append(score)
    return scores,np.average(scores)

def modelPlaySetOfGames(DQNModel, gamesToPlay=1,framesToAct=1, renderGame=False):
    _, average = modelPlay(DQNModel=DQNModel,gamesToPlay=gamesToPlay,renderGame=renderGame)
    return average

def qLearn(trainingDQN, targetDQN):
    # Add checkpoint manager for trainingModel
    # checkpointTrain = tf.train.Checkpoint(optimizer=optimizer, model=train_model)
    # managerTrain = tf.train.CheckpointManager(
    #     checkpointTrain, directory="./MsPacManQLearn/checkpoints/training", max_to_keep=5)

    # checkpointTarget = tf.train.Checkpoint(optimizer=optimizer, model=train_model)
    # managerTarget = tf.train.CheckpointManager(
    #     checkpointTarget, directory="./MsPacManQLearn/checkpoints/target", max_to_keep=5)

    #Create a file to hold stats about training
    # f = open('./MsPacManQLearn/trainingStatsData.csv','w')
    # f.write('Epoch, AI Game Avg Score, Loss,Epsilon,Training Ai Score\n')
    # f.close()

    # Set up q learning historical buffers
    prev_states = []
    prev_state_action = []
    next_states = []
    reward_history = []
    done_flags = []
    num_states_in_history = 200000
    train_model_after_num_actions = 4
    update_target_model_after_num_epochs = 10000
    
    # Book keeping variables
    frame_count = 0 # Number of frames seen
    epochs_ran = 0
    loss = 0

    trained = False
    while not trained:
        # Keras progress bar for training information
        #https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar
        prog_bar = tf.keras.utils.Progbar(NUM_STEPS_PER_EPOCH, width=30, verbose=1, interval=1.0, stateful_metrics=None, unit_name='step')

        loss_history = []
        epoch_reward = 0
        epoch_games_played = 1
        old_lives_left = 3

        # Create a queue to hold the last 4 frames of the game
        stacked_frames = FrameStack(FRAME_STACKING)
        stacked_state = stacked_frames.reset(preprocessFrame(env.reset(),FRAME_X_SIZE))
        stacked_state = stacked_state.reshape((INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))

        # state = env.reset()
        # state_tensor = tf.convert_to_tensor(state)
        # state_tensor = tf.expand_dims(state_tensor, 0)
        print("Epoch {0}/{1}".format(epochs_ran+1,NUM_EPOCHS))

        for i in range(1,NUM_STEPS_PER_EPOCH):
            frame_count += 1

            # Get action based on epsilion greedy algorithm
            # As we see more frames we will explore less and exploit more
            #epsilonGreedyState = stacked_state.reshape((1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))
            #epsilonGreedyState = stacked_state.reshape(1,stacked_state.shape[0],stacked_state.shape[1],stacked_state.shape[2])

            global epsilon
            if epsilon > random.random():
                action =  trainingDQN.exploreAction()
            else:
                action = trainingDQN.exploitAction(stacked_state)

            # Learn about it here
            # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
            # Calcualte new epsilion value
            # We want epsilion to slowly go down by decrementing a small portion each time
            epsilon -= (eps_max-eps_min)/eps_decay_steps
            epsilon = max(eps_min, epsilon) # Decaying policy with more steps

            # Do the action
            state_next_single,reward,done,lives_left = env.step(action)

            # Processes game frame
            # Add new state to the stacked frames
            # Will pop out the oldest frame
            stacked_state_next = stacked_frames.step(preprocessFrame(state_next_single,FRAME_X_SIZE))
            stacked_state_next = stacked_state_next.reshape((INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))

            #env.render()

            # Reset the enviornment if we lose all lives
            # Otherwise the game stays in this position for the remainder of the epoch
            # This was found out the hard way after 2 days of training and the bot
            # would stay in the same position, wonder why...
            if done == True:
                env.reset()
                #stacked_state = stacked_frames.reset(preprocessFrame(env.reset()[0],FRAME_X_SIZE))
                epoch_games_played+=1
                old_lives_left = 3
            elif lives_left['ale.lives'] != old_lives_left:
                # We want to penalize the bot for dying
                done = True
                old_lives_left = lives_left['ale.lives']

            # Update the historical lists
            prev_states.append(stacked_state)
            prev_state_action.append(action)
            next_states.append(stacked_state_next)
            reward_history.append(reward)
            done_flags.append(done)
            epoch_reward += reward

            # Update the current state
            stacked_state = stacked_state_next      

            # temp3 = stacked_state #stacked_state.reshape(INPUT_FRAME_SIZE[2],INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1])
            # for i in range(4):
            #     plt.figure(i)
            #     plt.imshow(temp3) 
                
            # plt.show()  # display it


            # Train the training model if we moved enough times
            if frame_count % train_model_after_num_actions == 0 and len(reward_history) > BATCH_SIZE:

                batch_states,batch_actions,batch_next_states,batch_rewards,batch_done = getBatchGameData(prev_states,prev_state_action,next_states,reward_history,done_flags,BATCH_SIZE)

                callBacks = trainingDQN.fit(target_model=targetDQN.getModel(),dcf=DISCOUNT_FACTOR,
                                            states=batch_states,actions=batch_actions,
                                            rewards=batch_rewards,next_states=batch_next_states,
                                            done=batch_done,batch_size=BATCH_SIZE)

                loss = callBacks.history['loss'][0]

            # If we have done enought epochs then we should update the target model
            if frame_count % update_target_model_after_num_epochs == 0:
                targetDQN.setWeights(trainingDQN.getModel())

            # Clear out ram for the buffer if we reach max length
            if len(reward_history) > num_states_in_history:
                #print("Deleting history")
                del prev_states[:1]
                del prev_state_action[:1]
                del next_states[:1]
                del reward_history[:1]
                del done_flags[:1]

            # Update kera progress bar
            prog_bar.update(i,values=[('loss',loss)])

        # End the progress bar to show the final numbers
        prog_bar.update(i+1,values=[('loss',loss)],finalize=True)

        # Save model after each epoch
        #managerTarget.save()
        #managerTrain.save()

        # # Run test games to check progress
        # numGameToRun = 3
        # #_,gameAverage = modelPlay(model=trainingModel,gamesToPlay=numGameToRun,renderGame=True)

        # scores = []
        # for _ in range(numGameToRun):
        #     # Create frame stacking
        #     stacked_frames = FrameStack(FRAME_STACKING)
        #     stacked_state = stacked_frames.reset(preprocessFrame(env.reset(),FRAME_X_SIZE))

        #     done = False
        #     score = 0
        #     frame_count = 0
        #     action = 0
        #     while not done:
        #         frame_count +=1
                
        #         #Get new state, reward, and if we are done
        #         # if frame_count % FRAME_STACKING == 0:
        #         #     temp = stacked_state.reshape(1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2])
        #         #     action = exploitAction(model,temp)

        #         temp = stacked_state.reshape(1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2])
        #         #action = exploitAction(trainingModel,temp)
        #         action = trainingDQN.exploitAction(temp)

        #         state_next_single,reward2,done,lives_left = env.step(action)
        #         stacked_state = stacked_frames.step(preprocessFrame(state_next_single,FRAME_X_SIZE))

        #         env.render()

        #         score += reward2

        #     scores.append(score)


        # print("{0} game average: {1:0.2f}".format(numGameToRun,np.average(scores)))
        #print("{0} game average: {1:0.2f}".format(numGameToRun,gameAverage))
        print("Epsilon: {:0.5f}".format(epsilon))
        print("Epoch reward {:0.0f}".format(epoch_reward/epoch_games_played))

        #loss_avg = np.average(loss_history)

        # Save data for a graph
        # f = open('./MsPacManQLearn/trainingStatsData.csv','a')
        # f.write('{0},{1},{2},{3},{4}\n'.format(epochs_ran,gameAverage,loss_avg,epsilon,epoch_reward))
        # f.close()

        # Need to come up with break condition
        epochs_ran += 1
        if epochs_ran == NUM_EPOCHS:
            trained = True
           
def main():
    
    target_model = DQNModel()
    train_model = DQNModel()

    target_model.createModel(inputDataSize=INPUT_FRAME_SIZE,numActions=possibleMovesLen,lr=LEARNING_RATE,name="Target")
    train_model.createModel(inputDataSize=INPUT_FRAME_SIZE,numActions=possibleMovesLen,lr=LEARNING_RATE,name="Train")

    target_model.setWeights(train_model.getModel())

    train_model.getModel().summary()

    qLearn(trainingDQN=train_model,targetDQN=target_model)

    #train_model.save('./MsPacManQLearn/savedModels/firstTry/train')
    #target_model.save('./MsPacManQLearn/savedModels/firstTry/target')
    
    # print("Testing models")
    # testingScores, average = modelPlay(target_model,100,renderGame=False)

    # print("Average score for {0} games: {1}".format(100,average))
    # print("Scores for all games")
    # print(testingScores)

    input("Press enter to see AI play game")
    testingScores, average = modelPlay(target_model,3,renderGame=True)
    

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()