from collections import deque
import numpy as np

# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

class FrameStack():
    def __init__(self,numStackFrames):
        self.frames = deque([],numStackFrames)
        self.numStackedFrames = numStackFrames

    def reset(self,frame):
        # Combine the input frame to find the max
        maxFrame = np.maximum(frame,frame)

        # Fill the stack states with initial state
        for _ in range(self.numStackedFrames):
            self.frames.append(maxFrame)

        #return np.stack(self.frames, axis=2)
        return np.array(self.frames)
    
    def step(self,frame):
        # Take the max of the last frame and the current on to extract different features
        maxFrame = np.maximum(self.frames[-1],frame)


        self.frames.append(maxFrame)

        # Return the current state and the list of states
        #return np.stack(self.frames, axis=2)
        return np.array(self.frames)



