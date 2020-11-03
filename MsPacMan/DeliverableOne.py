import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np
from entity import entity
from math import sqrt
import copy
import random
import argparse

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



def main(numIterations):
    # Make sure numIterations is positive
    if numIterations < 0:
        numIterations = 1

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

    #Create enviornemnt
    env = gym.make("MsPacman-v0",frameskip=2)

    #Give time for the render to open up
    env.render()
    time.sleep(2)

    framesBeforeRandom = 2
    framesBeforeInput = 2
    scores = list()
    for i in range(numIterations):
        # Need to reset enviornment before running
        env.reset()

        observation,reward,done,_ = env.step(4)  

        # This loop is for the begining sequence where the starting tune is played
        # We just skip past it
        for q in range(100):
            observation,reward,done,_ = env.step(3)
            env.render()

        # Vars for keeping track of game
        framesSinceLastInput = 0
        framesSinceLastMove = 0
        lastAction = 0
        score = 0
        action = 4
        
        # Need a previous to check if we are stuck
        pastPlayer = player
        
        for i in range(1000):
        #while not done:
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

                        # Check if we are on the same x as the closest ghost
                        # If so we want to move up or down to get away from it
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

                        # Figure out if the closest ghost is above/below or left/right of us
                        elif abs(deltaX) > abs(deltaY):
                            # If below us we want to go up
                            if deltaX > 0:
                                #print("UP")
                                action = 5
                            else:
                                # If above us we want to go up
                                #print("DOWN")
                                action = 2
                        else:
                            # If to the right of us we want to go left
                            if deltaY < 0:
                                #print("LEFT")
                                action = 3
                                
                            else:
                                # If to the right of us we want to go right
                                #print("RIGHT")
                                action = 4
                                
            
            # Save the players new location
            pastPlayer = copy.deepcopy(player)    
                
            # Get new state, reward, and if we are done
            observation,reward,done,_ = env.step(action)

            score += reward
            lastAction = action
            framesSinceLastInput += 1

            # Display the game
            env.render()

            

        print("Run {0}: {1}".format(i,int(score)))
        scores.append(score)
    
    if numIterations > 1:
        print("Average score: {0}".format(int(np.average(scores))))

# Only run code if main called this file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',type=int,default=1)
    args = parser.parse_args()

    numIterations = int(args.e)

    main(numIterations)