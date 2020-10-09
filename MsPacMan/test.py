#from gym.envs.atari import AtariEnv
import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np

def findSpriteLocation(image,ghost_color,ghost_size):
    for i in range(0,image.shape[0],4):
        for q in range(image.shape[1]):
            if np.array_equal(image[i][q],ghost_color):
                return (i,q)

    return (-1,-1)

def main():

    ghost_width = 8
    ghost_height = 10
    ghost_size = [ghost_width,ghost_height]
    
    pink_ghost = [198,89,179]
    red_ghost = [200,72,72]
    blue_ghost = [84,184,153]
    orng_ghost = [180,122,48]

    player_size = [7,7]
    player = [210,164,74]

    ghost_location = list()

    

    #game = AtariEnv(game='ms_pacman', obs_type='image')
    #game.render()

    env = gym.make("MsPacman-v0")

    #Give time for the render to open up
    env.render()
    time.sleep(2)

    env.reset()

    # Get an action
    action = env.action_space.sample()

    # Get new state, reward, and if we are done
    observation,reward,done,_ = env.step(action)

    bg_color = observation[2]
    postProcFrame = np.copy(observation)

    temp = findSpriteLocation(observation,red_ghost,ghost_size)
    print("Red ghost around {0}".format(temp))

    temp = findSpriteLocation(observation,player,player_size)
    print("MsPacman around {0}".format(temp))

    for pixel in postProcFrame:
        pixel = np.absolute(pixel - bg_color)
    #processedFrame = np.absolute(observation[:,:,2] - observation[:,:,0])



    #viewer = rendering.SimpleImageViewer()
    #viewer.imshow(processedFrame)

    # Reset environment for new state
    for i in range(1000):
        env.step(action)
        env.render()
        time.sleep(1/60)

# Only run code if main called this file
if __name__ == "__main__":
    main()

