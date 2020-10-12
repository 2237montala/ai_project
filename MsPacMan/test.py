#from gym.envs.atari import AtariEnv
import gym
import time
from gym.envs.classic_control import rendering
import cv2
import numpy as np
from entity import entity
from Tree import Tree
import copy

def findSpriteLocation(image,sprite):
    for i in range(0,image.shape[0],4):
        for q in range(image.shape[1]):
            if np.array_equal(image[i][q],sprite.color):
                return (i,q)

    return (-1,-1)

def findPossiblePaths(image,player_entity,block_entity,backgound):
    # Create a tree to hold the paths found
    root = Tree(player_entity.location)

    root.data = player_entity.location
    # Recusivly find open paths
    findOpenPaths(image,player_entity,block_entity,backgound,root)

    return root

def findOpenPaths(image,player_entity,block_entity,backgound,treeNode):
    # Make a copy of the input node so we can mess with it
    treeNodeCopy = treeNode

    # Find next block to the left
    # Find the first backgound color item, could be a block
    # Starting point is the players mid point
    theoreicalPlayer = copy.deepcopy(player_entity.location)
    playerMid = copy.deepcopy(player_entity.location)
    playerMid[0] += int(player_entity.size[0]/2)

    

    #print(playerMid)

    

    # The starting position should be right where the block is on ms pacmans sprite
    # This means we need to move the position down by 3
    #playerMid[0] += 3

    blocksInThisPath = 0    
    wallFound = False
    while not wallFound:
        # Figure out where the first occurance of the background color is
        # Or if there is a path way on any side of the player
        # The valid player area is 12 pixels tall
        while not np.array_equal(image[playerMid[0]+1][playerMid[1]],block_entity.color):
            leftUpEdge = [theoreicalPlayer[0] - 2, theoreicalPlayer[1] - 3]
            rightUpEdge = copy.deepcopy(leftUpEdge)
            rightUpEdge[1] += 11
            # Only checking the edges of the possible open path
            # Check if there is a path upwards
            if np.array_equal(image[leftUpEdge[0]][leftUpEdge[1]], backgound) and \
                np.array_equal(image[rightUpEdge[0]][rightUpEdge[1]], backgound):
                # Some opening above us
                print("Up path at player location: [{0},{1}]".format(theoreicalPlayer[0],theoreicalPlayer[1]))

            # # Check if there is a path downwards
            # if image[playerMid[0]][playerMid[1]-8] == backgound:
            #     # Some opening below us
            #     pass
            # # Check if there is a path left
            # if image[playerMid[0]][playerMid[1]] == backgound:
            #     # Some opening above us
            #     pass
            

            # Move the starting y value over 1
            theoreicalPlayer[1] -=1
            playerMid[1] -=1


        # We know we found either a wall or a block
        # Move back one pixel so we can test what it is
        playerMid[1] += 1
        theoreicalPlayer[1] += 1

        # A wall is a area of background color that has no blue above or below it
        # A block is a 4 x 2 area of background color with blue around it
        
        # Get a 6 x 4 size area to check for blocks with surrounding blue
        print(playerMid)
        area = getPixelSample((4,6,3),playerMid,block_entity,image)        
        
        # Check if the area sampled is a wall or not
        if isWall(area,block_entity):
            wallFound = True
        else:
            # Add this block to the list of nodes
            # Make a copy of the startPoint as it is a object and changes
            copyplayerMid = copy.deepcopy(playerMid)

            # Add the valid spot to the tree
            treeNodeCopy.addChild(copyplayerMid,'Left')
            treeNodeCopy = treeNodeCopy.getChild('Left')

            # Go double the length of a block to the left
            #playerMid[1] -= block_entity.size[0]*2
            # playerMid[1] -=2
            # theoreicalPlayer[1] += 2
            pixelsSinceLastBlock = 8
            blocksInThisPath+=1

        # Go in increments of block size
        #print(area)
        #print("")
        
    #treeNode = treeNodeCopy
    print("Blocks in this path: {0}".format(blocksInThisPath))

def getPixelSample(pixelArea,startingPoint,block_entity,gameFrame):
    # Make sure the user passed in a tuple
    assert(isinstance(pixelArea, tuple))

    area = np.zeros(pixelArea,dtype=int)
    for i in range(0,block_entity.size[1]+2):
            x = startingPoint[0] + i
            start = startingPoint[1] - area.shape[1]+1
            end = startingPoint[1]+1
            #print("Starting at point {2},{0} to {2},{1}".format(start,end,x))
            area[i][0:6] = gameFrame[x][start:end]

    return area

def isWall(sampleArea,block_entity):
    # Check edges for blue
    error = 0
    # Top of box
    for x in sampleArea[0]:
        if np.array_equal(x,block_entity.color):
            error += 1
    # Bottom of box
    for x in sampleArea[-1]:
        if np.array_equal(x,block_entity.color):
            error += 1
    # Left side
    for i in range(sampleArea.shape[0]):
        if np.array_equal(sampleArea[i][0],block_entity.color):
            error += 1
    # Right side
    for i in range(sampleArea.shape[0]):
        if np.array_equal(sampleArea[i][-1],block_entity.color):
            error += 1
    return error > 0

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

    block = entity(block_color,block_size)    

    bg_color = [0,28,136]

    #game = AtariEnv(game='ms_pacman', obs_type='image')
    #game.render()

    env = gym.make("MsPacman-v0")

    #Give time for the render to open up
    env.render()
    #time.sleep(2)

    env.reset()

    # Get an action
    action = env.action_space.sample()

    # Get new state, reward, and if we are done
    observation,reward,done,_ = env.step(action)

    red_ghost.location = list(findSpriteLocation(observation,red_ghost))
    print("Red ghost around {0}".format(red_ghost.location))

    player.location = list(findSpriteLocation(observation,player))
    # Subtact 1 from x cause there is a blue square in ms pac mans head
    # And id rather her location be at a corner of her sprite
    player.location[0] -= 1
    print("MsPacman around {0}".format(player.location))

    findPossiblePaths(observation,player,block,bg_color)
    #processedFrame = np.absolute(observation[:,:,2] - observation[:,:,0])



    #viewer = rendering.SimpleImageViewer()
    #viewer.imshow(processedFrame)

    # Reset environment for new state
    # for i in range(1000):
    #     env.step(action)
    #     env.render()
    #     time.sleep(1/60)

# Only run code if main called this file
if __name__ == "__main__":
    main()

