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
import time

# More references
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c

# Create game
env = gym.make("MsPacman-v0")

#possibleMoves = [2,3,4,5]
possibleMoves = [0,1,2,3,4,5,6,7,8]
possibleMovesLen = len(possibleMoves)


FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160
FRAME_REDUCTION = 2
FRAME_STACKING = 4
INPUT_FRAME_SIZE = (int(FRAME_X_SIZE/FRAME_REDUCTION),int(FRAME_Y_SIZE/FRAME_REDUCTION),FRAME_STACKING)

# NUM_EPOCHS = 300 # Changes how many times we run q learning
# FRAME_COUNT_MAX = 3000000
NUM_STEPS_PER_EPOCH = 10000 # How many frames will be ran through each epoch
BATCH_SIZE = 32   # Number of games per training session
LEARNING_RATE = 0.00025 
DISCOUNT_FACTOR = 0.99 # How much the current state reward is reduced by

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 1000000.0 # this value specifies how many frame we need to see before
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
                time.sleep(0.033)

            score += reward

        scores.append(score)
    return scores,np.average(scores)

def modelPlaySetOfGames(DQNModel, gamesToPlay=1,framesToAct=1, renderGame=False):
    _, average = modelPlay(DQNModel=DQNModel,gamesToPlay=gamesToPlay,renderGame=renderGame)
    return average

def qLearn(trainingDQN, targetDQN):
    checkpointTarget = tf.train.Checkpoint(optimizer=targetDQN.getModel().optimizer, model=targetDQN.getModel())
    managerTarget = tf.train.CheckpointManager(
        checkpointTarget, directory="checkpoints/target", max_to_keep=5)

    #Create a file to hold stats about training
    # f = open('trainingStatsData.csv','w')
    # f.write('Epoch, Loss,Epsilon,Training Ai Score\n')
    # f.close()

    # Set up q learning historical buffers
    prev_states = []
    prev_state_action = []
    next_states = []
    reward_history = []
    done_flags = []
    episode_reward_history = []
    num_states_in_history = 100000
    train_model_after_num_actions = 4
    update_target_model_after_num_frames = 10000
    
    # Book keeping variables
    frame_count = 0 # Number of frames seen
    epochs_ran = 0
    running_reward = 0
    loss = 0

    epsilon = 1.0

    trained = False
    while not trained:
        loss_history = []
        epoch_reward = 0
        epoch_games_played = 1
        old_lives_left = 3

        # Create a queue to hold the last 4 frames of the game
        stacked_frames = FrameStack(FRAME_STACKING)
        stacked_state = stacked_frames.reset(preprocessFrame(env.reset(),FRAME_X_SIZE))
        stacked_state = stacked_state.reshape((INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))

        #print("Epoch {0}".format(epochs_ran+1))

        for _ in range(1,NUM_STEPS_PER_EPOCH):
            frame_count += 1

            # Get action based on epsilion greedy algorithm
            # As we see more frames we will explore less and exploit more
            # Model expects a (1,85,80,4) array so we need to add a dimension to the image
            greedyEpsilonState = stacked_state.reshape((1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))
            if epsilon > random.random():
                action =  trainingDQN.exploreAction()
            else:
                action = trainingDQN.exploitAction(greedyEpsilonState)

            # Learn about it here
            # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
            # Calcualte new epsilion value
            # We want epsilion to slowly go down by decrementing a small portion each time
            epsilon -= (eps_max-eps_min)/eps_decay_steps
            epsilon = max(eps_min, epsilon) # Decaying policy with more steps

            # Do the action for 4 consecutive frames
            # This is modeled afer Env wrapper MaxandSkip
            # This fixes the frame maxing issue from before
            max_4_frames = []
            reward_4_frames = 0
            done = False
            for _ in range(FRAME_STACKING):
                state_next_single,reward,done,lives_left = env.step(action)

                max_4_frames.append(preprocessFrame(state_next_single,FRAME_X_SIZE))
                reward_4_frames += reward

                if done:
                    break

            # Max the last 4 frames to remove any disappearing sprites
            state_next_single = np.array(max_4_frames).max(axis=0)
            reward = reward_4_frames

            # Processes game frame
            # Add new state to the stacked frames
            # Will pop out the oldest frame
            stacked_state_next = stacked_frames.step(state_next_single)
            stacked_state_next = stacked_state_next.reshape((INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))

            #env.render()

            # Reset the enviornment if we lose all lives
            # Otherwise the game stays in this position for the remainder of the epoch
            # This was found out the hard way after 2 days of training and the bot
            # would stay in the same position, wonder why...
            if done == True:
                old_lives_left = 3
            elif lives_left['ale.lives'] != old_lives_left:
                # We want to penalize the bot for dying
                done = True
                old_lives_left = lives_left['ale.lives']

            # Save Score before clipping
            epoch_reward += reward

            # Reward cliping
            # Deepmind talks about this in there paper
            # Helps the bot understand death
            reward = np.sign(reward)

            # Update the historical lists
            prev_states.append(stacked_state)
            prev_state_action.append(action)
            next_states.append(stacked_state_next)
            reward_history.append(reward)
            done_flags.append(done)
            
            # Update the current state
            stacked_state = stacked_state_next      

            # Train the training model if we moved enough times
            if frame_count % train_model_after_num_actions == 0 and len(reward_history) > BATCH_SIZE:

                batch_states,batch_actions,batch_next_states,batch_rewards,batch_done = getBatchGameData(prev_states,prev_state_action,next_states,reward_history,done_flags,BATCH_SIZE)

                callBacks = trainingDQN.fit(target_model=targetDQN.getModel(),dcf=DISCOUNT_FACTOR,
                                            states=batch_states,actions=batch_actions,
                                            rewards=batch_rewards,next_states=batch_next_states,
                                            done=batch_done,batch_size=BATCH_SIZE)

                loss = callBacks.history['loss'][0]
                loss_history.append(loss)


            # If we have done enought epochs then we should update the target model
            if frame_count % update_target_model_after_num_frames == 0:
                targetDQN.setWeights(trainingDQN.getModel())

            # Delete the oldest value in the history buffers
            if len(reward_history) > num_states_in_history:
                del prev_states[:1]
                del prev_state_action[:1]
                del next_states[:1]
                del reward_history[:1]
                del done_flags[:1]

            # If we die we want to start over with a new game
            if done == True:
                break


        print("Epoch {0},Epsilon: {1:0.4f}, Epoch Reward {2:0.0f}, Frame Count {3}".format(epochs_ran,epsilon,(epoch_reward/epoch_games_played),frame_count))

        # Save data for a graph
        # loss_avg = np.average(loss_history)
        # f = open('trainingStatsData.csv','a')
        # f.write('{0},{1},{2},{3}\n'.format(epochs_ran,loss_avg,epsilon,epoch_reward))
        # f.close()


        epochs_ran += 1

        # Save model every 100 epochs
        if epochs_ran % 100 == 0:
            managerTarget.save()
           
def main():
    
    target_model = DQNModel()
    train_model = DQNModel()

    target_model.createModel(inputDataSize=INPUT_FRAME_SIZE,numActions=possibleMovesLen,lr=LEARNING_RATE,name="Target")
    train_model.createModel(inputDataSize=INPUT_FRAME_SIZE,numActions=possibleMovesLen,lr=LEARNING_RATE,name="Train")

    target_model.setWeights(train_model.getModel())

    train_model.getModel().summary()

    qLearn(trainingDQN=train_model,targetDQN=target_model)
    
    print("Testing models")
    testingScores, average = modelPlay(target_model,100,renderGame=False)

    print("Average score for {0} games: {1}".format(100,average))
    print("Scores for all games")
    print(testingScores)

    input("Press enter to see AI play game")
    _, average = modelPlay(target_model,10,renderGame=True)
    print("Average score for {0} games: {1}".format(10,average))
    

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()