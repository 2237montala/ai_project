import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np
from entity import entity
from math import sqrt
import copy
import random

def findSpriteLocation(image,sprite,boardSize):
    searchSize = 20

    if not np.array_equal(sprite.location,[-1,-1]):
        # If we have a past ghost location then well search around its known location
        # We create a box of 40x40 pixels to search
        # Make sure we done search outside the range of the game board
        xRange=np.array([sprite.location[0] - searchSize,sprite.location[0] + searchSize])
        yRange=np.array([sprite.location[1] - searchSize,sprite.location[1] + searchSize])

        np.clip(xRange,0,boardSize[0]-1,out=xRange)
        np.clip(yRange,0,boardSize[1]-1,out=yRange)

        #print("X range: {0},{1}".format(xRange[0],xRange[1]))
        #print("Y range: {0},{1}".format(yRange[0],yRange[1]))
        #print("why are you printing")

        # for i in range(xRange[0],xRange[1], 4):
        #     possibleLocations = np.where(sprite.color == image[i][yRange[0]:yRange[1]])
        #     if len(possibleLocations[0])>0:
        #         print([i,possibleLocations[0][0]])
        #         return [i,possibleLocations[0][0]]

        for i in range(xRange[0],xRange[1], 4):
            for q in range(yRange[0],yRange[1]):
                if np.array_equal(image[i][q],sprite.color):
                    return (i,q)

    # If we have no idea where the ghost is or the previous region search fails
    # then search the whole image for it
    temp = np.where(sprite.color==image)
    if len(temp[0])>0:
        return [temp[0][0],temp[1][0]]
    return [-1,-1]


def findClosestGhost(image,listOfGhosts,player_entity,boardSize):
    closestGhost = 100000
    whichGhostIsClose = -1
    playerLoc = player_entity.location
    for i in range(len(listOfGhosts)):
        tempGhost = findSpriteLocation(image,listOfGhosts[i],boardSize)
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
    board_size = (210,160)
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
    time.sleep(2)

    numIterations = 100
    scores = list()
    for i in range(numIterations):
        env.reset()

        observation,reward,done,_ = env.step(4)  

        for i in range(100):
            # Get new state, reward, and if we are done
            observation,reward,done,_ = env.step(3)
            env.render()
            #time.sleep(0.1)

        framesSinceLastInput = 0
        framesBeforeInput = 2
        action = 4
        framesSinceLastMove = 0
        framesBeforeRandom = 2
        pastPlayer = player
        lastAction = 0
        score = 0

        while not done:
            player.setLocation(findSpriteLocation(observation,player,board_size))
        
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
                    #print("Random")
                    action = randomAction(lastAction) 
                else:
                    # If not stuck then try to move away from the closest ghost
                    closestGhost,ghostDistance = findClosestGhost(observation,listOfGhosts,player,board_size)
                    #print(ghostDistance)

                    if ghostDistance < 40:
                        deltaX = player.location[0] - closestGhost.location[0]
                        deltaY = player.location[1] - closestGhost.location[1]

                        if 12 < abs(deltaX) > 0:
                                # Up or down?
                                # Randnum between 0 and 1
                                if random.randint(0,1) == 0:
                                    # Up
                                    #print("Same y: up")
                                    action = 5
                                else:
                                    # Down
                                    #print("Same y: Down")
                                    action = 2

                        elif abs(deltaX) > abs(deltaY):
                            if deltaX > 0:
                                #print("UP")
                                #action = env.action_space(2)
                                action = 5
                            else:
                                #print("DOWN")
                                #action = env.action_space(5)
                                action = 2
                        else:
                            if deltaY < 0:
                                
                                #print("LEFT")
                                #action = env.action_space(4)
                                action = 3
                                
                            else:
                                #print("RIGHT")
                                #action = env.action_space(3)
                                action = 4
                                
            
            # Save the players new location
            pastPlayer = copy.deepcopy(player)    
                
            # Get new state, reward, and if we are done
            observation,reward,done,_ = env.step(action)

            score += reward

            lastAction = action

            env.render()
            #time.sleep(0.1)

            framesSinceLastInput += 1

        print(score)
        scores[i] = score
        #input("Press enter to close window")
    
    print(np.average(scores))

# Only run code if main called this file
if __name__ == "__main__":
    main()