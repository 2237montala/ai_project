class entity():
    def __init__(self,color,size,location =[-1,-1]):
        self.color = color
        self.size = size
        self.location = location

    def setLocation(self,location=[-1,-1]):
        self.location = location