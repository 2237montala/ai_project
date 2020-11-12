import gym
import time
import cv2
import numpy as np
from entity import entity
from math import sqrt
import copy
import random

class OldAi():
    def __init__(self, boardSize):
        self.framesBeforeRandom = 2
        self.framesBeforeInput = 2

        self.board_size = (boardSize[0],boardSize[1])
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
        self.player = entity(player_color,player_size)

        
        red_ghost = entity(red_ghost_color,ghost_size)
        pink_ghost = entity(pink_ghost_color,ghost_size)
        blue_ghost = entity(blue_ghost_color,ghost_size)
        orng_ghost = entity(orng_ghost_color,ghost_size)

        self.listOfGhosts = list()
        self.listOfGhosts.append(red_ghost)
        self.listOfGhosts.append(pink_ghost)
        self.listOfGhosts.append(blue_ghost)
        self.listOfGhosts.append(orng_ghost)

        self.block = entity(block_color,block_size)    
        self.bg_color = [0,28,136]

    def findSpriteLocation(self,image,sprite,boardSize):
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


    def findClosestGhost(self,image,listOfGhosts,player_entity,boardSize):
        closestGhost = 100000
        whichGhostIsClose = -1
        playerLoc = player_entity.location
        for i in range(len(listOfGhosts)):
            tempGhost = self.findSpriteLocation(image,listOfGhosts[i],boardSize)
            listOfGhosts[i].setLocation(tempGhost)

            if tempGhost[0] >= 0:
                tempGhostDist = sqrt((tempGhost[0]-playerLoc[0])**2 + (tempGhost[1]-playerLoc[1])**2)

                if tempGhostDist < closestGhost:
                    closestGhost = tempGhostDist
                    whichGhostIsClose = i

        return listOfGhosts[whichGhostIsClose],closestGhost

    def randomAction(self,lastAction):
        actions = [2,3,4,5]

        if lastAction > 0:
            # Dont do the same action twice
            actions.remove(lastAction)

        return random.choice(actions)


    def runGame(self, gameEnv, gameSteps):
        gameFrames = []
        # This loop is for the begining sequence where the starting tune is played
        # We just skip past it
        for q in range(100):
            observation,reward,done,_ = gameEnv.step(3)
            #env.render()

        # Vars for keeping track of game
        framesSinceLastInput = 0
        framesSinceLastMove = 0
        lastAction = 0
        score = 0
        action = 4
        
        # Need a previous to check if we are stuck
        pastPlayer = self.player
        

        for i in range(gameSteps):
            self.player.setLocation(self.findSpriteLocation(observation,self.player,self.board_size))
        
            # Check if player is stuck
            if self.player.location == pastPlayer.location:
                # Compare the past location to the current
                framesSinceLastMove +=1
            else:
                framesSinceLastMove = 0

            if framesSinceLastInput == self.framesBeforeInput:
                framesSinceLastInput = 0

                # If the player is stuck then do a random action
                if framesSinceLastMove >= self.framesBeforeRandom:
                    #print("Random")
                    action = self.randomAction(lastAction) 
                else:
                    # If not stuck then try to move away from the closest ghost
                    closestGhost,ghostDistance = self.findClosestGhost(observation,self.listOfGhosts,self.player,self.board_size)
                    #print(ghostDistance)

                    if ghostDistance < 40:
                        deltaX = self.player.location[0] - closestGhost.location[0]
                        deltaY = self.player.location[1] - closestGhost.location[1]

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
            pastPlayer = copy.deepcopy(self.player)    
                
            if i > 0:
                gameFrames.append([observation,action])

            # Get new state, reward, and if we are done
            observation,reward,done,_ = gameEnv.step(action)

            score += reward
            lastAction = action
            framesSinceLastInput += 1

            if done:
                break

            # Display the game
            #gameEnv.render()

        return score,gameFrames
