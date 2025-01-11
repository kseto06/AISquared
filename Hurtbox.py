import typing
from dataclasses import dataclass
from Cube import Cube

@dataclass
class Hurtbox:
    x: float
    y: float
    size_x: float
    size_y: float
    x_max: float = 0.0 # max = current pos + size
    y_max: float = 0.0
    
    # def __init__(self):
    #     self.x -= self.size_x/2
    #     self.y -= self.size_y/2
    #     self.x_max += self.size_x
    #     self.y_max += self.size_y

    def update_pos(self, x: float, y: float) -> None:
        '''
        This function takes in an updated set of positions (x, y) and updates the original

        Inputs: 
        x: float 
        y: float
        '''
        self.x -= self.size_x/2
        self.y -= self.size_y/2
        self.x_max += self.size_x
        self.y_max += self.size_y

    def hurtbox(self, body1: Cube, body2: Cube) -> bool:
        '''
        This function takes in two bodies

        Inputs:
        other_obj: Cube (The other agent object)
        '''

        # Get the bounding boxes of the bodies
        bb1 = body1.shape.cache_bb() 
        bb2 = body2.shape.cache_bb() 

        # Check for overlap
        return bb1.intersects(bb2)
        





