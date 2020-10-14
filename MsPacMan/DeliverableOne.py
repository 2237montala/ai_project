import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np
from entity import entity
from math import sqrt
import copy
import random

def findSpriteLocation(image,sprite):
    for i in range(0,image.shape[0],4):
        for q in range(image.shape[1]):
            if np.array_equal(image[i][q],sprite.color):
                return (i,q)

    return (-1,-1)

def findClosestGhost(image,listOfGhosts,player_entity):
    closestGhost = 100000
    whichGhostIsClose = -1
    playerLoc = player_entity.location
    for i in range(len(listOfGhosts)):
        tempGhost = findSpriteLocation(image,listOfGhosts[i])
        listOfGhosts[i].setLocation(tempGhost)

        if tempGhost[0] >= 0:
            tempGhostDist = sqrt((tempGhost[0]-playerLoc[0])**2 + (tempGhost[1]-playerLoc[1])**2)

            if tempGhostDist < closestGhost:
                closestGhost = tempGhostDist
                whichGhostIsClose = i

    return listOfGhosts[whichGhostIsClose],closestGhost

def randomAction(lastAction):
    actions = [2,3,4,5]

    if lastAction > 0:
        # Dont do the same action twice
        actions.remove(lastAction)

    return random.choice(actions)



def main():
    ghost_width = 8
    ghost_height = 10
    ghost_size = [ghost_width,ghost_height]

    block_size = [4,2]
    block_color = [228,111,111]
    
    pink_ghost_color = [198,89,179]
    red_ghost_color = [200,72,72]
    blue_ghost_color = [84,184,153]
    orng_ghost_color = [180,122,48]

    player_size = [7,10]
    player_color = [210,164,74]
    player = entity(player_color,player_size)

    
    red_ghost = entity(red_ghost_color,ghost_size)
    pink_ghost = entity(pink_ghost_color,ghost_size)
    blue_ghost = entity(blue_ghost_color,ghost_size)
    orng_ghost = entity(orng_ghost_color,ghost_size)

    listOfGhosts = list()
    listOfGhosts.append(red_ghost)
    listOfGhosts.append(pink_ghost)
    listOfGhosts.append(blue_ghost)
    listOfGhosts.append(orng_ghost)

    block = entity(block_color,block_size)    

    bg_color = [0,28,136]

    random.seed(time.time)

    env = gym.make("MsPacman-v0",frameskip=2)

    #Give time for the render to open up
    env.render()
    #time.sleep(2)

    env.reset()

    observation,reward,done,_ = env.step(4)

    

    for i in range(90):
        # Get new state, reward, and if we are done
        observation,reward,done,_ = env.step(3)
        env.render()
        #time.sleep(0.1)

    framesSinceLastInput = 0
    framesBeforeInput = 4
    action = 4
    framesSinceLastMove = 0
    framesBeforeRandom = 2
    pastPlayer = player
    lastAction = 0

    while not done:
        player.setLocation(findSpriteLocation(observation,player))
    
        # Check if player is stuck
        if player.location == pastPlayer.location:
            # Compare the past location to the current
            framesSinceLastMove +=1
        else:
            framesSinceLastMove = 0

        if framesSinceLastInput == framesBeforeInput:
            framesSinceLastInput = 0

            # If the player is stuck then do a random action
            if framesSinceLastMove >= framesBeforeRandom:
                print("Random")
                action = randomAction(lastAction) 
            else:
                # If not stuck then try to move away from the closest ghost
                closestGhost,ghostDistance = findClosestGhost(observation,listOfGhosts,player)
                #print(ghostDistance)

                if ghostDistance < 30:
                    deltaX = player.location[0] - closestGhost.location[0]
                    deltaY = player.location[1] - closestGhost.location[1]

                    if 12 < abs(deltaX) > 0:
                            # Up or down?
                            # Randnum between 0 and 1
                            if random.randint(0,1) == 0:
                                # Up
                                print("Same y: up")
                                action = 2
                            else:
                                # Down
                                print("Same y: Down")
                                action = 5

                    elif abs(deltaX) > abs(deltaY):
                        if deltaX > 0:
                            print("UP")
                            #action = env.action_space(2)
                            action = 2
                        else:
                            print("DOWN")
                            #action = env.action_space(5)
                            action = 5
                    else:
                        if deltaY < 0:
                            
                            print("LEFT")
                            #action = env.action_space(4)
                            action = 3
                            
                        else:
                            print("RIGHT")
                            #action = env.action_space(3)
                            action = 4
                            
           
        # Save the players new location
        pastPlayer = copy.deepcopy(player)    
               
        # Get new state, reward, and if we are done
        observation,reward,done,_ = env.step(action)

        lastAction = action

        env.render()
        #time.sleep(0.1)

        framesSinceLastInput += 1

    input("Press enter to close window")

# Only run code if main called this file
if __name__ == "__main__":
    main()