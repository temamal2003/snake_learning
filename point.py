class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x,y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self.x)+str(self.y))