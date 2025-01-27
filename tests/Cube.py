import pymunk

class Cube:
    def __init__(self, position, space, mass, collision_type, width=50, height=10):
        self.mass = mass
        self.width = width
        self.height = height

        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape.body = self.body
        
        self.shape.collision_type = collision_type
        self.shape.body.position = position
        space.add(self.shape, self.body)