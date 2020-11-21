import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History

import random as random
from FrameStack import FrameStack

class DQNModel():
    def __init__(self):
        pass

    def createModel(self,inputDataSize,numActions,lr,name):

        self.inputSize = inputDataSize
        self.numActions = numActions

        inputs = layers.Input(shape=(85, 80, 4,))

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(128, activation="relu")(layer4)
        action = layers.Dense(numActions, activation="linear")(layer5)

        self.model = keras.models.Model(inputs=inputs, outputs=action, name=name)

        # Set up the optimizer function
        # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
        # He describes the hyper paramters at the bottom
        optimizer = optimizer=keras.optimizers.RMSprop(lr=lr, rho=0.95, epsilon=0.01, momentum=0.95)

        # # Set up loss function
        # loss_func = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        # https://openai.com/blog/openai-baselines-dqn/
        self.model.compile(optimizer, loss='huber')

    def setWeights(self, modelToCopyFrom):
        self.model.set_weights(modelToCopyFrom.get_weights())

    def getModel(self):
        return self.model

    def exploreAction(self):
        return np.random.choice(self.numActions)

    def exploitAction(self, state):
        
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)

        return tf.argmax(action_probs[0]).numpy()
        # predictions = self.model.predict(state)
        # return np.argmax(self.model(state,training=False))

    def fit(self,target_model,dcf,states,actions,rewards,next_states,done,batch_size):
        history = History()

        #action_one_hot = np.ones(actions.shape)
        possible_rewards = target_model.predict(next_states)
        #possible_rewards = target_model.predict([next_states, action_one_hot])

        # Debugging line for seeing if the done state is correct
        # if np.any(done[:] == 1):
        #     print()

        # Calculate best q value for each state
        q_values = rewards + (dcf * np.max(possible_rewards, axis=1))

        # If any state lead to a death then set it to -1
        updated_q_values = q_values * (1 - done) - done

        # Create a once hot array for the actions
        action_mask = tf.one_hot(actions, self.numActions,on_value=1.0,off_value=0.0)

        # train the model by multiplying the actions by the q values
        self.model.fit(states,action_mask * updated_q_values[:, None],batch_size=batch_size,
                        epochs=1,callbacks=[history],verbose=0)

        return history