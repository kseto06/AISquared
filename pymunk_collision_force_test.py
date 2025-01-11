import pymunk
import pygame
import gymnasium as gym
import random
import sys
from pymunk.pygame_util import DrawOptions

width = 600
height = 600

class Ball:
    def __init__(self, position, space, mass, collision_type, width=50, height=10):
        self.mass = mass
        self.width = width
        self.leg_height = 30
        self.height = height

        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape.body = self.body

        self.height = height + self.leg_height #Total height = height (of box) + height of leg segment
        leg1 = pymunk.Segment(self.body, (-20, 30), (-10, 0), 3)  # 2
        leg2 = pymunk.Segment(self.body, (20, 30), (10, 0), 3)
        
        self.shape.collision_type = collision_type
        self.shape.body.position = position
        space.add(self.shape, self.body, leg1, leg2)

class Ground:
    def __init__(self, space, x, y, width_ground):
        self.body = pymunk.Body(x, y, body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (width_ground-100, 10))
        self.shape.body.position = (width//2, 275+y)
        space.add(self.shape, self.body)

def collide(arbiter, space, data):
    print("DIE!!!!!!1")
    return True

def hitbox(entity1: Ball, entity2: Ball):
    '''
    This function detects when entity2 hits entity1.

    @param entity1  Entity to check for
    @param entity2  Reference entity 
    '''

    # Check hitboxes on the x:
    if entity1.shape.body.position.x <= entity2.shape.body.position.x and entity1.shape.body.position.y <= entity2.shape.body.position.y <= entity1.shape.body.position.y + entity1.height:
        if entity1.shape.body.position.x < entity2.shape.body.position.x and \
           entity1.shape.body.position.x + entity1.width > entity2.shape.body.position.x:
            print("Entity1 hit by Entity2 on the LEFT")
            return True
        elif entity2.shape.body.position.x < entity1.shape.body.position.x and \
             entity2.shape.body.position.x + entity2.width > entity1.shape.body.position.x:
            print("Entity1 hit by Entity2 on the RIGHT")
            return True

    # Case 1y: Entity1 below
    # if entity1.shape.body.position.y < entity2.shape.body.position.y and entity1.shape.body.position.x <= entity2.shape.body.position.x <= entity1.shape.body.position.x + entity1.width:
    #     if entity2.shape.body.position.y >= entity1.shape.body.position.y and entity2.shape.body.position.y <= entity1.shape.body.position.y + entity1.height:
    #         print("Entity1 hit by Entity2 from the TOP")
    #         return True
    # # Case 2y: Entity1 above
    # elif entity1.shape.body.position.y > entity2.shape.body.position.y and entity1.shape.body.position.x <= entity2.shape.body.position.x <= entity1.shape.body.position.x + entity1.width:
    #     if entity2.shape.body.position.y >= entity1.shape.body.position.y and entity2.shape.body.position.y <= entity1.shape.body.position.y + entity1.height:
    #         print("Entity1 hit by Entity2 from the BOTTOM")
    #         return True

    return False

def draw_bounding_box(screen, shape, color=(255, 0, 0)):
    '''
    This function draws the bounding box of a cube/rectangle in Pygame
    Inputs:
    screen (Pymunk screen)
    shape (shape Object to reference)
    color (RGB Color values of the bounding box)
    '''
    pygame.draw.rect(
        screen, color, 
        pygame.Rect(shape.shape.body.position.x-shape.width/2, shape.shape.body.position.y-shape.height/2, shape.width, shape.height), 1
    )

def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Collision Handler")
    clock = pygame.time.Clock()

    draw_options = DrawOptions(screen)

    space = pymunk.Space()
    space.gravity = 0, 681.0
    # x = random.randint(150, 380)
    x = 100
    ground = Ground(space, 25, 100, 585)
    platform1 = Ground(space, 25, -35, 340)
    ball = Ball((x, 100), space, 4, 1)
    ball2 = Ball((x+370, 100), space, 3, 2)
    collision_handler = space.add_collision_handler(1, 2)
    collision_handler.begin = collide

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)

        # Detect hitbox between balls
        hitbox(ball, ball2)

        """
        Set the original ball (ball) to be the pursuer
        """
        if ball2.shape.body.position.x < ball.shape.body.position.x and not 0 <= ball2.shape.body.position.x <= 100:
            
            ball.shape.body.apply_force_at_local_point((-150, 0), (0,0))
        elif ball2.shape.body.position.x < ball.shape.body.position.x and 0 <= ball2.shape.body.position.x <= 100:
            if ball.shape.body.velocity.x > 50:
                ball.shape.body.velocity = (0,0)
            else:
                ball.shape.body.apply_force_at_local_point((150, 0), (0,0))

        if ball2.shape.body.position.x > ball.shape.body.position.x and not 500 <= ball2.shape.body.position.x <= 600:
            
            ball.shape.body.apply_force_at_local_point((150, 0), (0,0))
        elif ball2.shape.body.position.x > ball.shape.body.position.x and 500 <= ball2.shape.body.position.x <= 600:
            if ball.shape.body.velocity.x > 50:
                ball.shape.body.velocity = (0, 0)
            else:
                ball.shape.body.apply_force_at_local_point((-150, 0), (0,0))

        # if ball.shape.body.position.x < ball2.shape.body.position.x:
        #     """  (0, 400) means apply 400 units of force in the direction of x
        #     (0,0) is the co-ordinate to apply the force to (should be applied to coordinate of floor )"""
        #     ball.shape.body.apply_force_at_local_point((50, 0), (0, 0))
        # else:
        #     ball.shape.body.apply_force_at_local_point((-50, 0), (0, 0))

        if ball.shape.body.position.y < ball2.shape.body.position.y:
            ball.shape.body.apply_force_at_local_point((0, -150), (0, 0))
        
        # Prevent excessive jumping
        # ball.shape.body.apply_force_at_local_point((0, 1500/2), (0, 0))

        # Y-direction (jump)
        # if int(ball.shape.body.position.y) == 237:
        #     """  (0, 400) means apply 400 units of force in the direction of y 
        #     (0,0) is the co-ordinate to apply the force to (should be applied to coordinate of floor )"""
        #     ball.shape.body.apply_force_at_local_point((0, -3500), (0, 0))

        # Escaping horizontally
        if ball2.shape.body.position.x < ball.shape.body.position.x and not 0 < ball2.shape.body.position.x < 100:
            ball2.shape.body.apply_force_at_local_point((-50, 0), (0,0))
        if ball2.shape.body.position.x > ball.shape.body.position.x and not 500 <= ball2.shape.body.position.x <= 600:
            ball2.shape.body.apply_force_at_local_point((50, 0), (0,0))
        # Panic rush - move towards the pursuer but try jumping away
        if ball2.shape.body.position.x < ball.shape.body.position.x and ball2.shape.body.position.x <= 100:
            if ball2.shape.body.velocity.y > 70:
                ball2.shape.body.velocity = (0, 0)
            else:
                ball2.shape.body.apply_force_at_local_point((350, -1500/2), (0,0))
        if ball2.shape.body.position.x > ball.shape.body.position.x and ball2.shape.body.position.x > 500:
            if ball2.shape.body.velocity.y > 70:
                ball2.shape.body.velocity = (0, 0)
            else:
                ball2.shape.body.apply_force_at_local_point((-350, -1500/2), (0,0))
            
        screen.fill((0, 0, 0))
        draw_bounding_box(screen, ball, color=(0, 255, 0))
        draw_bounding_box(screen, ball2, color=(255, 0, 0))
        space.debug_draw(draw_options)
        space.step(1/50.0)
        pygame.display.update()
        clock.tick(50)
        # print(ball.shape.body.velocity)


if __name__ == '__main__':
    sys.exit(main())