import pymunk
import pygame
import gymnasium as gym
import numpy as np
import sys
from pymunk.pygame_util import DrawOptions
from enum import Enum
from typing import Union

width = 600
height = 600

DELTA = 1/50.0

# Custom draw/rendering options for Pymunk objects
class CustomDrawOptions(DrawOptions):
    def __init__(self, screen):
        super().__init__(screen)

    def draw_circle(self, pos, angle, radius, outline_color=(0,0,0), fill_color=(255,255,255)):
        '''
        Custom draw options for Pymunk circle
        '''
        super().draw_circle(pos, angle, radius, outline_color, fill_color)

    def draw_circle(self, verts, radius, outline_color=(0,0,0), fill_color=(255,255,255)):
        '''
        Custom draw options for Pymunk polygon
        '''
        super().draw_polygon(verts, radius, outline_color, fill_color)

class GameObject:
    # ACTUAL IMAGE RENDERING: 
    def __init__(self, x: int, y: int, dir_name: str, screen: pygame.display):
        self.x = x
        self.y = y
        self.hitboxes = []
        self.image = pygame.image.load(dir_name)
        self.image_width, self.image_height = self.image.get_size()
        self.image_x = (x + self.image_width//1.5)
        self.image_y = (y + self.image_height//1.5) #Centered
        self.screen = screen

    def update_pos(self, x: int, y: int):
        self.x = x
        self.y = y
        self.image_x = (x + self.image_width//2) - 10
        self.image_y = (y + self.image_height//2) - 10#Centered

    # GAME OBJECT HITBOXES

    def add_hitbox(self, x: int, y: int, width: int, height: int):
        '''
        Append a new hitbox to the GameObject's collection of hitboxes
        '''
        self.hitboxes.append(Hitbox(x, y, width, height, self.screen))

    def draw_object(self):
        self.screen.blit(self.image, (self.image_x, self.image_y))

class Sword(GameObject):
    '''
    Sword (png) object. Extends game object
    '''
    def __init__(self, x: int, y: int, dir_name: str, screen):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y

    def construct_hitboxes(self):
        # Construct multiple hitboxes based on their (x, y) coordinates
        # NOTE: Change this for different types of game objects
        hitbox_points = [
            ((self.x + self.image_width//2) - 10, self.y), #initial (x, y)
            ((self.x+self.image_width) - 10, self.y),
            ((self.x+3*self.image_width//2) - 10, self.y)
        ]

        # Add hitboxes to the space using the hitbox_points
        L = len(hitbox_points)
        for i in range(L - 1):
            x1, y1 = hitbox_points[i][0], hitbox_points[i][1]
            x2, y2 = hitbox_points[i+1][0], hitbox_points[i+1][1]
            width = self.image_width // (L-1)
            height = self.image_height
            x = min(x1, x2)  # Determine the x-coordinate of the hitbox
            y = min(y1, y2)  # Determine the y-coordinate of the hitbox
            self.add_hitbox(x, y, width, height) 

    def update_hitbox_pos(self, x, y):
        self.x = x
        self.y = y
        self.construct_hitboxes()

    # TODO: Powercast a simple sword move, logic to make it appear/disappear
    def dash_attack():
        pass

class Throw(GameObject):
    pass


class Hammer(GameObject):
    """
    Hammer (png) object. Extends GameObject.
    """

    def __init__(self, x: int, y: int, dir_name: str, screen, player, opponent):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y
        self.player = player
        self.opponent = opponent
        self.cool_down_count = 0

        on_hit_power = Power(10,self.player,self.opponent,"""game_obj.y -= 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered
kick_impulse = (0, -50)  # Negative Y for an upward kick
self.opponent.body.apply_impulse_at_world_point(kick_impulse, self.opponent.body.position)  # Center of the ball
        """)
        on_miss_power = Power(10,self.player,self.opponent,"""game_obj.y -= 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered    
        """)
        initial_power = Power(
            10, self.player, self.opponent,
            power_script="""game_obj.y += 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered
        """,
            on_hit_power=on_hit_power,
            on_miss_power=on_miss_power
        )


        self.smash_attack = Move(initial_power, self.player, self.opponent)
       

    def construct_hitboxes(self):
        self.hitboxes = [] #martin edited
        self.add_hitbox(
            self.image_x , self.image_y + self.image_height/2 - 5/2,  # Top-left corner
            self.image_width-15,  # Width
            5 # Height
        )

        self.add_hitbox(
            self.image_x + (self.image_width-15), self.image_y ,  # Top-left corner
            15,  # Width
            self.image_height # Height
        )
        
    def update_hitbox_pos(self, x, y):
        self.x = x
        self.y = y
        self.construct_hitboxes()
        # Add hammer handle hitbox
       # self.hitboxes.append(Hitbox(self.x, self.y, self.image_width+10, self.image_height, self.screen))
       

'''
TODO: Add more object/action classes here as needed
'''

@DeprecationWarning #Initial 3b1b ball. Use the new cube object
class Ball:
    def __init__(self, position, space, mass, collision_type, width=50, height=10):
        self.mass = mass
        self.width = width
        self.height = height
        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape.body = self.body
        leg1 = pymunk.Segment(self.body, (-20, 30), (-10, 0), 3)  # 2
        leg2 = pymunk.Segment(self.body, (20, 30), (10, 0), 3)
        self.shape.collision_type = collision_type
        self.shape.body.position = position
        space.add(self.shape, self.body, leg1, leg2)

class Ground:
    def __init__(self, space: pymunk.Space, x: int, y: int, width_ground: int):
        self.body = pymunk.Body(x, y, body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (width_ground, 10))
        self.shape.body.position = (x + width_ground // 2, y)
        self.shape.friction = 0.8
        space.add(self.shape, self.body)

class Cube:
    def __init__(self, position: pymunk.Vec2d, space: pymunk.Space, mass: float, collision_type: int, screen: pygame.display, cube_color: tuple[int, int, int, int], state, platforms: list[Ground, Ground], width=50, height=10, health=0):
        self.mass = mass
        self.width = width
        self.height = height

        # Draw the cube/square shape
        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape.body = self.body

        self.shape.collision_type = collision_type
        self.shape.body.position = position
        space.add(self.shape, self.body)

        # Add eyes using Pymunk circles (to prevent obstruction):
        self.left_eye = pymunk.Circle(self.body, radius=8, offset=(-width // 4, -height // 5))
        self.right_eye = pymunk.Circle(self.body, radius=8, offset=(width // 4, -height // 5))
        self.left_pupil = pymunk.Circle(self.body, radius=4, offset=(-width // 4, -height // 5))
        self.right_pupil = pymunk.Circle(self.body, radius=4, offset=(width // 4, -height // 5))
        
        # Make eyes non-colliding
        self.left_eye.filter = pymunk.ShapeFilter(categories=0)
        self.right_eye.filter = pymunk.ShapeFilter(categories=0)
    
        # Add eyes to the space
        space.add(self.left_eye, self.right_eye, self.left_pupil, self.right_pupil)

        self.shape.color = cube_color
        self.left_eye.color = (255, 255, 255, 255)  
        self.left_pupil.color = (0, 0, 0, 255)
        self.right_eye.color = (255, 255, 255, 255)
        self.right_pupil.color = (0, 0, 0, 255)

        # Health
        self.health = health

        # Movement
        self.direction = 0
        self.move_speed = 50 #On the x
        self.jump_speed = 50

        # Start state = InAirState
        self.state = state

        # Platforms
        self.platforms = platforms

    def set_direction(self) -> int:
        '''
        Set the direction of an object [-1, 0, 1]; moving left, stationary, right
        '''
        # self.body.velocity = pymunk.Vec2d(self.body.velocity.x * dir, self.body.velocity.y)
        if self.body.velocity.x > 0:
            self.direction = 1
        elif self.body.velocity.x < 0:
            self.direction = -1
        else:
            self.direction = 0 #Stationary

    # def get_direction(self) -> int:
    #     if self.body.velocity.x > 0:
    #         return 1 #Right
    #     elif self.body.velocity.x < 0:
    #         return -1 #Left
    #     else:
    #         return 0 #Stationary
        
    def update_health(self, new_health: float):
        self.health += int(new_health)

    def get_bounding_box(self) -> pymunk.Shape.cache_bb:
        return self.shape.cache_bb()

    def is_on_floor(self):
        for platform in self.platforms:
            # Check if object is touching a platform
            if self.shape.cache_bb().intersects(platform.shape.cache_bb()):
                return True
        
        return False        
    
    def _physics_process(self, delta: float) -> None:
        new_state: PlayerState = self.state.physics_process(self, delta)

        if new_state != None:
            self.state.exit(self)
            print(self.state.get_state_name(), " -> ", new_state.get_state_name())
            self.state = new_state
            self.state.enter(self)
        
        # if self.state.action == 'light_attack':
        #     side_slash.start_cast()

        # move_and_slide()

# FINITE STATE MACHINE
class State(Enum):
    STANDING = 1
    JUMPING = 2
    ATTACKING = 3

class PlayerState:
    def __init__(self, action = None):
        self.action = action

    def get_state_name(self) -> str:
        return "PlayerState"
    
    def enter(self, player):
        pass

    def physics_process(self, player, delta): 
        return None
    
    def exit(self, player):
        pass

    def animate_player(self, player):
        pass

    def on_hurtbox_damaged(self, player):
        pass
    
    def set_new_action(self, action: str = None): 
        self.action = action

    def check_current_action(self) -> Union[str, None]:
        if self.action == 'jump':
            return 'jump'
        elif self.action == 'attack':
            return 'attack'
        else:
            return None
        
class GroundState(PlayerState):
    def get_state_name(self) -> str:
        return "GroundState"

    @staticmethod
    def get_ground_state(player: Cube) -> PlayerState:
        if player.direction:
            return WalkingState()
        else:
            return StandingState()
        
    def physics_process(self, player: Cube, delta: float) -> Union[PlayerState, None]:
        if player.direction:
            player.body.velocity = pymunk.Vec2d(player.direction * player.move_speed, player.body.velocity.y)
            #Physics already handled by Pymunk?
        else:
            # p.velocity.x = move_toward(p.velocity.x, 0, p.pop.move_speed)
            player.shape.friction = 0.01

        # Handling jump
        if self.check_current_action() == 'jump' and player.is_on_floor():
            player.body.velocity = pymunk.Vec2d(player.body.velocity.x, player.jump_speed)
            return InAirState()
        elif not player.is_on_floor():
            return InAirState()
        else:
            return None
            
class InAirState(PlayerState):
    def get_state_name(self) -> str:
        return "InAirState"
    
    def physics_process(self, player: Cube, delta: float) -> PlayerState:
        # gravity setting here
        if player.direction:
            player.body.velocity = pymunk.Vec2d(player.direction * player.move_speed, player.body.velocity.y)
        else:
            player.shape.friction = 0.01

        if player.is_on_floor():
            return GroundState.get_ground_state(player)
        else:
            return None
            
    def animate_player(self, player):
        pass #Jump should be handled by Pymunk

class HurtState(InAirState):
    def get_state_name(self) -> str:
        return "HurtState"
    
    def enter(player: Cube) -> None:
        player.body.velocity = pymunk.Vec2d(player.body.velocity.x, player.jump_speed)

    def physics_process(self, player: Cube, delta: float) -> PlayerState:
        if player.direction:
            player.body.velocity = pymunk.Vec2d(player.direction * player.move_speed, player.body.velocity.y)
        else:
            player.shape.friction = 0.01

        if player.is_on_floor():
            return GroundState.get_ground_state(player)
        else:
            return None
    
    def animate_player(self, player: Cube):
        pass

class WalkingState(GroundState):
    def get_state_name(self) -> str:
        return "WalkingState"
    
    def physics_process(self, player: Cube, delta: float) -> PlayerState:
        new_state: PlayerState = super().physics_process(player, delta)
        
        if player.direction or new_state:
            return new_state
        else:
            return StandingState()

class StandingState(GroundState):
    def get_state_name(self) -> str:
        return "StandingState"
    
    def physics_process(self, player: Cube, delta: float) -> PlayerState:
        new_state: PlayerState = super().physics_process(player, delta)

        if not player.direction or new_state:
            return new_state
        else:
            return WalkingState()
        
    def animate_player(self, player):
        pass

class Hitbox: 
    '''
    This class adds and renders a Hitbox to the game space.
    '''
    def __init__(self, x: int, y: int, width: int, height: int, screen: pygame.display):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = screen
        self.hitbox_rect = pygame.draw.rect(self.screen, (255, 0, 0), (self.x, self.y, self.width, self.height), 1)
    
    def draw(self):
        # pygame.draw.rect(self.screen, (255, 0, 0), (self.x, self.y, self.width, self.height), 1)
        self.hitbox_rect.topleft = (self.x, self.y)

class HitboxHandler:
    '''
    This class handles the hitbox interactions between Pygame (GameObject) and Pymunk (agent) objects.
    '''
    def __init__(self, screen: pygame.display):
        self.screen = screen

    def object_hits_agent(self, game_obj: GameObject, agent: Cube) -> bool:
        game_obj_hitboxes = game_obj.hitboxes
        agent_bb = agent.get_bounding_box()

        # Bounding box rectangle for agent
        pygame.draw.rect(self.screen, (0, 255, 0), (int(agent_bb.left), int(agent_bb.bottom), int(agent_bb.right - agent_bb.left), int(agent_bb.top - agent_bb.bottom)), 1)
        
        for hitbox in game_obj_hitboxes:
            # Draw a rect for the game object
            game_obj_hitbox = pygame.draw.rect(self.screen, (255, 0, 0), (game_obj.x, game_obj.y, game_obj.image_width, game_obj.image_height), 1)

            # Check for overlap between a pygame.rect and pymunk bb
            if game_obj_hitbox.colliderect(agent_bb.left, agent_bb.bottom, agent_bb.right - agent_bb.left, agent_bb.top - agent_bb.bottom):
                print("hit")
                agent.update_health(3) # Take damage
                return True 

            return False

class Hurtbox:
    def hurtbox(self, body1: Cube, body2: Cube) -> bool:
        '''
        This function takes in two bodies and checks if their bounding boxes overlap 

        Inputs:
        body1 (Cube): first object
        body2 (Cube): second object
        '''

        # Get the bounding boxes of the bodies
        bb1 = body1.shape.cache_bb() 
        bb2 = body2.shape.cache_bb() 

        # Check for overlap
        return bb1.intersects(bb2)
    

# UI & GAME LOGIC
class UI:
    def __init__(self, screen: pygame.display, agent_1: Cube, agent_2: Cube):
        self.screen = screen
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        # Construct a list of agents
        self.agents = np.array([self.agent_1, self.agent_2])

    def display_UI(self):
        self.display_agent_healths()
        self.display_percentages()

    def display_agent_healths(self):
        # Left health UI
        points_left = np.array([(100, 100), (180, 80), (200, 160), (120, 180)])#topleft, topright, bottomright, bottomleft
        RED = (255, 0, 0)
        ORANGE = (255, 140, 0)
        BLUE = (0, 0, 255)
        GRAY = (128, 128, 128)
        WHITE = (255, 255, 255)

        # Tilted cube
        pygame.draw.polygon(self.screen, ORANGE, points_left-50) 

        # Agent name + textbox
        pygame.draw.rect(self.screen, GRAY, (points_left[0][0], points_left[0][1], 64*2, 20), 0) 
        self.draw_eyes(points_left[0][0]-25, points_left[0][1]-30, 10) # Left eye
        self.draw_eyes(points_left[0][0]+17.5, points_left[0][1]-40, 10) # Right eye
        font = pygame.font.Font(None, 19)
        # render text:
        text_surface = font.render("Agent 1", True, WHITE)
        text_rect = text_surface.get_rect(center=pygame.Rect(points_left[0][0], points_left[0][1], 128, 20).center)
        self.screen.blit(text_surface, text_rect)

        # Right health UI:
        points_right: np.ndarray = points_left
        for i in range(len(points_left)):
            points_right[i][0] += 300

        # Tilted cube
        pygame.draw.polygon(self.screen, BLUE, points_right-50) 

        # Agent name + textbox
        pygame.draw.rect(self.screen, GRAY, (points_right[0][0], points_right[0][1], 64*2, 20), 0) 
        self.draw_eyes(points_right[0][0]-25, points_right[0][1]-30, 10) # Left eye
        self.draw_eyes(points_right[0][0]+17.5, points_right[0][1]-40, 10) # Right eye
        font = pygame.font.Font(None, 19)
        # render text:
        text_surface = font.render("Agent 2", True, WHITE)
        text_rect = text_surface.get_rect(center=pygame.Rect(points_right[0][0], points_right[0][1], 128, 20).center)
        self.screen.blit(text_surface, text_rect)

    # Percentages (like SSBU)
    def display_percentages(self):
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        YELLOW = (255, 255, 0)
        DARK_RED = (139, 0, 0)

        # Agent percentage text
        font = pygame.font.Font(None, 55)
        # render text & text colours:
        for i in range(len(self.agents)):
            COLOUR = WHITE
            if 30 < self.agents[i].health < 70:
                COLOUR = YELLOW 
            elif 70 <= self.agents[i].health < 110: 
                COLOUR = RED
            elif self.agents[i].health >= 110:
                COLOUR = DARK_RED

            text_surface = font.render(f'{self.agents[i].health}.0%', True, COLOUR)
            text_rect = text_surface.get_rect(center=pygame.Rect(160+i*300, 50, 64, 50).center)
            self.screen.blit(text_surface, text_rect)

    def draw_eyes(self, x: float, y: float, radius: int):
        '''
        This function draws eyes
        '''
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        # Draw the white circle (outer part of the eye)
        pygame.draw.circle(self.screen, WHITE, (x, y), radius)
        pygame.draw.circle(self.screen, BLACK, (x, y), radius//2)


def collide(arbiter, space, data):
    # print("Collision detected!")
    return True

def display_video(width: int, height: int):
    import cv2
    import os
    import numpy as np
    import skvideo.io
    from IPython.display import Video

    frame_folder = "frames"
    frames = [os.path.join(frame_folder, f) for f in sorted(os.listdir(frame_folder)) if f.endswith(".png")]

    frame_array = []
    for frame_path in frames:
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        frame_array.append(img)

    video_data = np.stack(frame_array, axis=0)
    skvideo.io.vwrite('output.mp4', video_data)
    Video('output.mp4', embed=True, width=width, height=height)

def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Env")
    clock = pygame.time.Clock()

    draw_options = DrawOptions(screen)

    space = pymunk.Space()
    space.gravity = 0, 200.0
    
    # Modify the placement of the platforms
    ground = Ground(space, 50, 500, 500)
    platform1 = Ground(space, 125, 400, 125)  # First platform
    platform2 = Ground(space, 350, 400, 125)  # Second platform with a gap

    state1: PlayerState = InAirState()
    state2: PlayerState = InAirState()

    ball = Cube((150, 100), space, 4, 1, screen, cube_color=(255, 140, 0, 255), state=state1, platforms=[ground, platform1, platform2], width=50, height=50)
    sword = Sword(150, 100, "./assets/sword.png", screen)

    ball2 = Cube((450, 100), space, 3, 2, screen, cube_color=(0, 0, 255, 255), state=state2, platforms=[ground, platform1, platform2], width=50, height=50)
    hurtbox = Hurtbox()

    # Display initial UI
    ui = UI(screen=screen, agent_1=ball, agent_2=ball2)

    hitbox_handler = HitboxHandler(screen)
    collision_handler = space.add_collision_handler(1, 2)
    collision_handler.begin = collide
    frame_count = 0

    # Define the y-coordinate threshold for quitting
    quit_y_threshold = 575

    def apply_force_toward_each_other(ball1, ball2, force_magnitude):
        # Calculate force directions
        direction1 = (1 if ball2.shape.body.position.x > ball1.shape.body.position.x else -1, 0)
        direction2 = (1 if ball1.shape.body.position.x > ball2.shape.body.position.x else -1, 0)

        # Apply forces
        ball1.shape.body.apply_force_at_local_point((force_magnitude * direction1[0], force_magnitude * direction1[1]), (0, 0))
        ball2.shape.body.apply_force_at_local_point((force_magnitude * direction2[0], force_magnitude * direction2[1]), (0, 0))

    for _ in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit(0)

        hurtbox.hurtbox(ball, ball2)

        # Continuously move the balls toward each other
        # apply_force_toward_each_other(ball, ball2, 50)

        # Watch for FSM and physics_processes
        # print(ball.body.velocity, " ", ball2.body.velocity)
        ball.direction = 1
        ball2.direction = -1
        ball._physics_process(DELTA)
        ball2._physics_process(DELTA)

        # Quit if any ball goes below the threshold
        if ball.shape.body.position.y > quit_y_threshold or ball2.shape.body.position.y > quit_y_threshold:
            print("A ball has fallen below the threshold. Exiting...")
            sys.exit()

        screen.fill((0, 0, 0))

        # Draw GameObject & sword hitboxes:
        sword.update_pos(ball.shape.body.position.x, ball.shape.body.position.y)
        sword.update_hitbox_pos(ball.shape.body.position.x, ball.shape.body.position.y)
        for hitbox in sword.hitboxes:
            hitbox.draw()
        sword.draw_object()

        # Check if the GameObject hits Agent
        hitbox_handler.object_hits_agent(game_obj=sword, agent=ball2)

        # Render health UI
        ui.display_UI()

        # Draw the red line
        pygame.draw.line(screen, (255, 0, 0), (0, quit_y_threshold), (width, quit_y_threshold), 2)

        space.debug_draw(draw_options)
        space.step(DELTA)
        pygame.display.flip() # pygame.display.update()
        clock.tick(50)
        # pygame.image.save(screen, f"frames/frame_{frame_count:04d}.png")
        frame_count += 1

if __name__ == '__main__':
    main()