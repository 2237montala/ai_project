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

# More references
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c

# Create game
env = gym.make("MsPacman-v0",frameskip=4)

#                UP DOWN LEFT RIGHT
possibleMoves = [2   ,5  ,3   ,4]
possibleMovesLen = len(possibleMoves)

FRAME_X_SIZE = 170
FRAME_Y_SIZE = 160
FRAME_REDUCTION = 2
FRAME_STACKING = 4
INPUT_FRAME_SIZE = (int(FRAME_X_SIZE/FRAME_REDUCTION),int(FRAME_Y_SIZE/FRAME_REDUCTION),FRAME_STACKING)

NUM_EPOCHS = 120 # Changes how many times we run q learning
NUM_STEPS_PER_EPOCH = 10000 # How many frames will be ran through each epoch
BATCH_SIZE = 32   # Number of games per training session
LEARNING_RATE = 0.001 
DISCOUNT_FACTOR = 0.99 # How much the current state reward is reduced by

epsilon = 1.0  # Current epsilon values
eps_min = 0.05  
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

def createNetwork(inputDataSize,name):

    inputs = layers.Input(shape=inputDataSize)
    layer1 = layers.Conv2D(32,(8,8),4, activation='relu')(inputs)
    layer2 = layers.Conv2D(64,(4,4),2, activation='relu')(layer1)
    layer3 = layers.Conv2D(64,(3,3),1, activation='relu')(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(128, activation='relu')(layer4)
    action = layers.Dense(possibleMovesLen, activation='linear')(layer5)

    return keras.Model(inputs=inputs,outputs=action,name=name)

def exploreAction():
    return np.random.choice(possibleMoves)

def exploitAction(networkModel, state):
    return getAction(networkModel.predict(state))
    #return np.argmax(networkModel(state,training=False))

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

def modelPlay(model, gamesToPlay=1,framesToAct=1, renderGame=False):
    scores = []
    for i in range(gamesToPlay):
        #print("Game {0}".format(i+1))
        observation = env.reset()
        done = False

        score = 0
        frame_count = 0
        action = 0
        while not done:
            frame_count +=1
            if renderGame:
                env.render()

            #Get new state, reward, and if we are done
            if frame_count % framesToAct == 0:
                state = preprocessFrame(observation,FRAME_X_SIZE)
                state = state.reshape(1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1])

                #action = np.argmax(model.predict(state))
                action = getAction(model.predict(state))

            observation,reward,done,_ = env.step(action)
            
            score += reward

        scores.append(score)
    return scores,np.average(scores)

def modelPlaySetOfGames(model, gamesToPlay=1,framesToAct=1, renderGame=False):
    _, average = modelPlay(model,gamesToPlay,framesToAct=framesToAct,renderGame=renderGame)
    return average

def qLearn(trainingModel, targetModel):
    # Set up the optimizer function
    # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    # He describes the hyper paramters at the bottom
    optimizer = optimizer=keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01, momentum=0.95)

    # This was a guess taken from the keras website
    # This didn't work
    #optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    # # Set up loss function
    # loss_func = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    # https://openai.com/blog/openai-baselines-dqn/
    loss_func = keras.losses.Huber()

    # Add checkpoint manager for trainingModel
    checkpointTrain = tf.train.Checkpoint(optimizer=optimizer, model=trainingModel)
    managerTrain = tf.train.CheckpointManager(
        checkpointTrain, directory="./MsPacManQLearn/checkpoints/training", max_to_keep=5)

    checkpointTarget = tf.train.Checkpoint(optimizer=optimizer, model=targetModel)
    managerTarget = tf.train.CheckpointManager(
        checkpointTarget, directory="./MsPacManQLearn/checkpoints/target", max_to_keep=5)

    #Create a file to hold stats about training
    f = open('./MsPacManQLearn/trainingStatsData.csv','w')
    f.write('Epoch, AI Game Avg Score, Loss,Epsilon,Training Ai Score\n')
    f.close()

    # Set up q learning historical buffers
    prev_states = []
    prev_state_action = []
    next_states = []
    reward_history = []
    done_flags = []
    num_states_in_history = 1000000
    train_model_after_num_actions = 4
    update_target_model_after_num_epochs = NUM_STEPS_PER_EPOCH-1
    
    # Book keeping variables
    frame_count = 0 # Number of frames seen
    epochs_ran = 0
    loss = 0

    trained = False
    while not trained:
        # Keras progress bar for training information
        #https://www.tensorflow.org/api_docs/python/tf/keras/utils/Progbar
        prog_bar = tf.keras.utils.Progbar(NUM_STEPS_PER_EPOCH, width=30, verbose=1, interval=1.0, stateful_metrics=None, unit_name='step')

        state = env.reset()
        state = preprocessFrame(state,FRAME_X_SIZE)

        loss_history = []
        epoch_reward = 0
        epoch_games_played = 1
        old_lives_left = 3

        # Create a queue to hold the last 4 frames of the game
        stacked_frames = FrameStack(FRAME_STACKING)
        stacked_state.reset(state)

        print("Epoch {0}/{1}".format(epochs_ran+1,NUM_EPOCHS))

        for i in range(1,NUM_STEPS_PER_EPOCH):
            frame_count += 1

            # Get action based on epsilion greedy algorithm
            # As we see more frames we will explore less and exploit more
            #epsilonGreedyState = state.reshape(1,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1])

            global epsilon
            if random.random() < epsilon:
                action =  exploreAction()
            else:
                action = exploitAction(trainingModel,stacked_state)

            # Learn about it here
            # https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
            # Calcualte new epsilion value
            # We want epsilion to slowly go down by decrementing a small portion each time
            epsilon -= (eps_max-eps_min)/eps_decay_steps
            epsilon = max(eps_min, epsilon) # Decaying policy with more steps

            # Do the action
            state_next,reward,done,lives_left = env.step(action)

            #Processes game frame
            state_next = preprocessFrame(state_next,FRAME_X_SIZE)

            # Add new state to the stacked frames
            # Will pop out the oldest frame
            stacked_state.step(state)

            #env.render()

            # Reset the enviornment if we lose all lives
            # Otherwise the game stays in this position for the remainder of the epoch
            # This was found out the hard way after 2 days of training and the bot
            # would stay in the same position, wonder why...
            if done == True:
                env.reset()
                epoch_games_played+=1
                old_lives_left = 3
            elif lives_left['ale.lives'] != old_lives_left:
                # We want to penalize the bot for dying
                done = True
                old_lives_left = lives_left['ale.lives']

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
            if frame_count % train_model_after_num_actions == 0 and len(reward_history) > BATCH_SIZE:
                batch_states,batch_actions,batch_next_states,batch_rewards,batch_done = getBatchGameData(prev_states,prev_state_action,next_states,reward_history,done_flags,BATCH_SIZE)

                batch_next_states = batch_next_states.reshape((BATCH_SIZE,INPUT_FRAME_SIZE[0],INPUT_FRAME_SIZE[1],INPUT_FRAME_SIZE[2]))

                # Generate all the rewards for the next states
                possible_rewards = targetModel.predict(batch_next_states)  

                # Set a any terminal states to 0
                possible_rewards[batch_done] = -1

                # Generate q values from all the next states
                # Need to use tf.reduce_max because the possible_rewards is a tensor not an numpy array
                # Multiplies the best reward value by the discount factor. THis is the bellman equation
                q_values = batch_rewards + (DISCOUNT_FACTOR * np.max(possible_rewards, axis=1))
                #q_values = batch_rewards + (DISCOUNT_FACTOR * tf.reduce_max(possible_rewards, axis=1))

                # We need to penalize a death
                # Check each index in batch_done for if the game is over
                # for j in range(len(batch_done)):
                #     if batch_done[j] == True:
                #         q_values[j] = -1
                # Code above doesn't work as you can't edit a tensor using list manipulation
                # IDK how to fix
                # https://keras.io/examples/rl/deep_q_network_breakout/
                # What this line does is for each q value multiply it by (1 - each batch_done) and subtract batch_done
                # Batch done contains booleans so 1 - true = 0 so if the game ended at that frame then 
                # set that q value to -1 to punish the ai for making the move 
                #q_values = q_values * (1 - batch_done) - batch_done

                
                # Create a mask so that we can ignore q values that didn't change
                # Makes a one hot vector where each action is represented in binary
                # https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
                action_mask = tf.one_hot(batch_actions, possibleMovesLen,on_value=1.0,off_value=0.0,)

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

                loss_history.append(loss)

            # If we have done enought epochs then we should update the target model
            if frame_count % update_target_model_after_num_epochs == 0:
                # trainingModel.save('tmp_model')
                # targetModel = keras.models.load_model('tmp_model')
                temp = trainingModel.get_weights()
                temp2 = targetModel.get_weights()
                targetModel.set_weights(trainingModel.get_weights())
                temp2 = targetModel.get_weights()

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
        managerTarget.save()
        managerTrain.save()

        # Run test games to check progress
        numGameToRun = 3
        gameAverage = modelPlaySetOfGames(targetModel,numGameToRun,train_model_after_num_actions,renderGame=False)
        print("{0} game average: {1:0.2f}".format(numGameToRun,gameAverage))
        print("Epsilon: {:0.5f}".format(epsilon))
        print("Epoch reward {:0.0f}".format(epoch_reward/epoch_games_played))

        loss_avg = np.average(loss_history)

        # Save data for a graph
        f = open('./MsPacManQLearn/trainingStatsData.csv','a')
        f.write('{0},{1},{2},{3},{4}\n'.format(epochs_ran,gameAverage,loss_avg,epsilon,epoch_reward))
        f.close()

        # Need to come up with break condition
        epochs_ran += 1
        if epochs_ran == NUM_EPOCHS:
            trained = True
           

        

def main():
    # Create training model
    train_model = createNetwork(inputDataSize=INPUT_FRAME_SIZE,name="Train")

    # Create target model
    target_model = createNetwork(inputDataSize=INPUT_FRAME_SIZE,name="Target")

    train_model.summary()

    qLearn(trainingModel=train_model,targetModel=target_model)

    train_model.save('./MsPacManQLearn/savedModels/firstTry/train')
    target_model.save('./MsPacManQLearn/savedModels/firstTry/target')
    
    print("Testing models")
    testingScores, average = modelPlay(target_model,100,4,renderGame=False)

    print("Average score for {0} games: {1}".format(100,average))
    print("Scores for all games")
    print(testingScores)

    input("Press enter to see AI play game")
    testingScores, average = modelPlay(target_model,3,renderGame=True)
    

# Only run code if main called this file
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    main()