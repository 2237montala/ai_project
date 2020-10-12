class Tree:
    def __init__(self, data):
        self.children = [None,None,None,None]
        self.data = data

    def addChild(self,data,direction):
        if direction == 'Up':
            self.children[0] = Tree(data)
        elif direction == 'Right':
            self.children[1] = Tree(data)
        elif direction == 'Down':
            self.children[2] = Tree(data)
        elif direction == 'Left':
            self.children[3] = Tree(data)

    def getChild(self,direction):
        if direction == 'Up':
            return self.children[0]
        elif direction == 'Right':
            return self.children[1]
        elif direction == 'Down':
            return self.children[2]
        elif direction == 'Left':
            return self.children[3]