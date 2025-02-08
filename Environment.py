import pymunk
import pygame
import gymnasium as gym
import numpy as np
import sys
from pymunk.pygame_util import DrawOptions
from enum import Enum
from typing import Union
from PIL import Image, ImageSequence

width = 600
height = 600

DELTA = 1/50.0
CUBE_DIM = 50

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
        self.frames: Union[list, None] = None
        self.going_left = False
        self.screen = screen

    def get_object_name(self):
        # print(self.__class__.__name__)
        return self.__class__.__name__

    def update_pos(self, x: int, y: int):
        
        self.x = x
        self.y = y
        if self.going_left:
            self.image_x = (self.x + self.image_width//2) - 10 - CUBE_DIM*10
            self.image_y = (self.y + self.image_height//2) - 10
        else:
            self.image_x = (x + self.image_width//2) - 10
            self.image_y = (y + self.image_height//2) - 10#Centered
        

    # GAME OBJECT HITBOXES

    def add_hitbox(self, x: int, y: int, width: int, height: int):
        #Append a new hitbox to the GameObject's collection of hitboxes
        self.hitboxes.append(Hitbox(x, y, width, height, self.screen))
    
    def reflect_hitboxes(self):
        
        new_hitboxes = []
        for hitbox in self.hitboxes:
            # x = (self.x + CUBE_DIM/2) - (hitbox.x - (self.x + CUBE_DIM/2))
            x = 2 * (self.x + CUBE_DIM / 2) - hitbox.x - CUBE_DIM*2.5
            # y = (self.y + CUBE_DIM/2) - (hitbox.y - (self.y + CUBE_DIM/2))
            y = hitbox.y
            width = hitbox.width
            height = hitbox.height
            new_hitboxes.append(Hitbox(x, y, width, height, self.screen))
        self.hitboxes = new_hitboxes

    def reflect_object(self):
        
        if self.going_left:
            self.image = pygame.transform.flip(self.image, True, False)

    def update_hitbox_pos(self, x: float, y: float, going_left: bool):
        self.x = x
        self.y = y
        self.construct_hitboxes()
        self.going_left = going_left

        # Booleans to track positions
        if(self.going_left == True):
            self.reflect_hitboxes()
            # try:
            #     self.process()
            # except Exception as e:
            #     pass
            self.reflect_object(self.going_left)

    def process(self):
        self.animation_timer += self.animation_speed
        if self.animation_timer >= 1:
            self.animation_timer = 0
            self.current_frame_index += 1
            if self.current_frame_index >= len(self.frames):
                self.finished = True

    def draw_object(self):
        img: pygame.image = self.image

        if self.frames is not None:
            if isinstance(self.frames, list):
                img = self.frames[self.current_frame_index]

        self.screen.blit(img, (self.image_x, self.image_y))
    
    """ def reflect_hitboxes(self):
        
        new_hitboxes = []
        for hitbox in self.hitboxes:
            x = (self.x + CUBE_DIM/2) - (hitbox.x - (self.x + CUBE_DIM/2))
            y = (self.y + CUBE_DIM/2) - (hitbox.y - (self.y + CUBE_DIM/2))
            width = hitbox.width
            height = hitbox.height
            new_hitboxes.append(Hitbox(x, y, width, height, self.screen))
        self.hitboxes = new_hitboxes
    """
    
class Move:
    # A move manages powers and transitions between them.
    def __init__(self, initial_power, player, opponent, frames: list, draw_object: callable):
        self.initial_power = initial_power
        self.current_power = self.initial_power
        self.player = player
        self.opponent = opponent
        self.active = False

        self.hit_occurred = False

        self.cool_down_duration = 50 # can change to a parameter ~!

        self.frames = frames # List of animation frames
        self.logged_direction = None
        self.draw_object = draw_object
        self.logged_direction = -1
    def activate(self):
        if(not self.player.cool_down):
            self.active = True

    def execute(self, game_obj: GameObject, hitbox_handler):
        # if game_obj.get_object_name() == "GroundPound":
        #     print(self.current_power.frames != None, self.current_power.frame_count)
      #  if(self.player.direction != self.logged_direction):
       #     game_obj.reflect_hitboxes()
     #   self.logged_direction = self.player.direction
        if self.active == True and self.current_power:
            # self.draw_object(self.current_power.frames, frame_idx=self.current_power.frame_count)
            self.current_power.execute(game_obj)
           
          
            self.hit_occurred = hitbox_handler.object_hits_agent(game_objects=self.player.attacks, attacking_agent=self.player, attacked_agent=self.opponent)
            
            # Transition immediately on hit - NOTE: This may need to be removed since it will automatically trigger the attack
            print("opponent state action: "+ str(self.opponent.state.action))
           # if self.hit_occurred and self.current_power.on_hit_power:
              #  self.current_power = self.current_power.on_hit_power
           #     self.current_power.frame_count = 0
            #    self.current_power.execute(game_obj)

            # Transition on duration
            if self.current_power.is_finished():
                self.current_power.frame_count = 0

                next_power = self.current_power.transition(self.hit_occurred)
                if next_power is not self.current_power:
                    self.current_power = next_power

                if not self.hit_occurred:
                    #implies miss
                    self.player.cool_down = True
                    self.player.cool_down_duration = self.cool_down_duration
        else:
            self.active = False
            self.current_power = self.initial_power
      
    
class Power:
    def __init__(self, duration: float, player, opponent, frames: list, draw_object: callable, power_script: str="", on_hit_power=None, on_miss_power=None):
        self.duration = duration # duration is in frames (as well)
        self.frame_count = 0

        self.player = player
        self.opponent = opponent

        self.power_script = power_script

        self.on_hit_power = on_hit_power  # Power to transition to on hit
        self.on_miss_power = on_miss_power  # Power to transition to on miss

        self.frames = frames
        self.draw_object = draw_object

    def update_frame_count(self):
        self.frame_count += 1

        if(self.player.cool_down):
            self.player.cool_down_count += 1

    def execute(self, game_obj):
        exec(self.power_script)
        self.update_frame_count()
        # For GIFs:
        print(self.frames == None)
        if self.frames is not None:
            self.draw_object(self.frames, frame_idx=self.frame_count)

    def is_finished(self):
       # Check if all casts in the power have completed, or if the opponent got hit by any hitbox/cast during the attack
        return self.frame_count > self.duration

    def transition(self, hit_occurred):
        # Determine which power to transition to based on hit or miss.
        next_power = self.on_hit_power if hit_occurred else self.on_miss_power
        # if next_power is None:
        #     return self
        
        # if next_power.frames is None:
        #     next_power.frames = self.frames
        return next_power


class Sword(GameObject):
    '''
    Sword (png) object. Extends game object
    '''
    def __init__(self, x: int, y: int, dir_name: str, screen):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y
        self.attack = None
    def construct_hitboxes(self):
       
        self.hitboxes = []  # Ensure old hitboxes are cleared first
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

class Throw(GameObject):
    pass

class Hammer(GameObject):

    def __init__(self, x: int, y: int, dir_name: str, screen, player, opponent):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y
        self.player = player
        self.opponent = opponent

        on_hit_power = Power(10,self.player,self.opponent, None, None, """game_obj.y -= 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered
kick_impulse = (0, -50)  # Negative Y for an upward kick
self.opponent.body.apply_impulse_at_world_point(kick_impulse, self.opponent.body.position+(0, -10))  # Center of the ball
        """)
        on_miss_power = Power(10,self.player,self.opponent, None, None, """game_obj.y -= 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered    
        """)
        initial_power = Power(
            10, self.player, self.opponent, None, None,
            power_script="""game_obj.y += 5
game_obj.image_x = (game_obj.x + game_obj.image_width//2) - 10
game_obj.image_y = (game_obj.y + game_obj.image_height//2) - 10#Centered
        """,
            on_hit_power=on_hit_power,
            on_miss_power=on_miss_power
        )

        self.attack = Move(initial_power, self.player, self.opponent, None, None)
       

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
        
    #def update_hitbox_pos(self, x, y):
       # self.x = x
       # self.y = y
       # self.construct_hitboxes()
        # Add hammer handle hitbox
       # self.hitboxes.append(Hitbox(self.x, self.y, self.image_width+10, self.image_height, self.screen))
       
    """
withint the hammer class
on_hit_power = Power(10,"")
on_miss_power = Power(10,"")
initial_power = Power(10,"",on_hit_power, on_miss_power)

"""

    # TODO: Add specific hammer attack logic


class Punch(GameObject):

    def __init__(self, x: int, y: int, dir_name: str, screen, player, opponent):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y
        self.player = player
        self.opponent = opponent
 
        self.delta_x = 0 #x displacement due to punch
        self.delta_y = 0 #y displacement due to punch

        on_hit_power = Power(5,self.player,self.opponent, None, None, """
game_obj.delta_x += 5
KICK_FORCE = 150
kick_impulse = self.opponent.body.position - (game_obj.x,game_obj.y)
kick_impulse = kick_impulse/kick_impulse.length * KICK_FORCE
self.opponent.body.apply_impulse_at_world_point(kick_impulse, self.opponent.body.position+(0, -10))  # Center of the ball
        """)
        on_miss_power = Power(5,self.player,self.opponent,None, None, """
game_obj.delta_x += 5
        """)
        initial_power = Power(5, self.player, self.opponent, None, None,
            power_script="""
game_obj.delta_x -= 5
        """,
            on_hit_power=on_hit_power,
            on_miss_power=on_miss_power
        )

        self.attack = Move(initial_power, self.player, self.opponent, None, None)
    
    def update_pos(self, x: int, y: int):
        self.x = x
        self.y = y

        self.image_x = (x + self.image_width//2) - 10 -75 +10 + self.delta_x
        self.image_y = (y + self.image_height//2) - 10 -20+10 + self.delta_y#Centered

    def construct_hitboxes(self):
        self.hitboxes = [] #martin edited
        self.add_hitbox(
            self.image_x , self.image_y,  # Top-left corner
            self.image_width,  # Width
            self.image_height # Height
        )
        
 #   def update_hitbox_pos(self, x, y):
   ##     self.x = x 
  #      self.y = y
  #      self.construct_hitboxes()
        # Add hammer handle hitbox
       # self.hitboxes.append(Hitbox(self.x, self.y, self.image_width+10, self.image_height, self.screen))
       
    """
withint the hammer class
on_hit_power = Power(10,"")
on_miss_power = Power(10,"")
initial_power = Power(10,"",on_hit_power, on_miss_power)

"""

class GroundPound(GameObject):
    def __init__(self, x: int, y: int, dir_name: str, screen: pygame.display, player, opponent):
        super().__init__(x, y, dir_name, screen)
        self.x = x
        self.y = y
        self.player = player
        self.opponent = opponent

        #Gif:
        scale = 1
        gif = Image.open(dir_name)
        self.frames = [
            pygame.transform.scale(
                pygame.image.fromstring(frame.convert("RGBA").tobytes(), frame.size, "RGBA"),
                (int(frame.width * scale), int(frame.height * scale))
            )
            for frame in ImageSequence.Iterator(gif)
        ]
        self.image_width /= scale
        self.image_height /= scale
   
        self.animation_timer = 0
        self.animation_speed = 0.1
        self.current_frame_index = 0
        self.finished = False  # Whether the animation is done
        self.delta_x = 0 #x displacement due to punch
        self.delta_y = 0 #y displacement due to punch
        self.going_left = False

        on_hit_power = Power(5,self.player,self.opponent,self.frames,self.draw_object,"""
game_obj.delta_x += 5
KICK_FORCE = 150
kick_impulse = self.opponent.body.position - (game_obj.x,game_obj.y)
kick_impulse = kick_impulse/kick_impulse.length * KICK_FORCE
self.opponent.body.apply_impulse_at_world_point(kick_impulse, self.opponent.body.position+(0, -10))  # Center of the ball
        """)
        on_miss_power = Power(5,self.player,self.opponent,self.frames,self.draw_object,"""
game_obj.delta_x += 5
        """)
        initial_power = Power(5, self.player, self.opponent,self.frames,self.draw_object,
            power_script="""
game_obj.delta_x -= 5
        """,
            on_hit_power=on_hit_power,
            on_miss_power=on_miss_power
        )
        
        self.attack = Move(initial_power, self.player, self.opponent, self.frames, self.draw_object)
    
    # @Override
    def reflect_hitboxes(self):
        new_hitboxes = []
        for hitbox in self.hitboxes:
            # x = (self.x + CUBE_DIM/2) - (hitbox.x - (self.x + CUBE_DIM/2))
            x = 2 * (self.x + CUBE_DIM / 2) - hitbox.x - CUBE_DIM*2.5
            # y = (self.y + CUBE_DIM/2) - (hitbox.y - (self.y + CUBE_DIM/2))
            y = hitbox.y
            width = hitbox.width
            height = hitbox.height
            new_hitboxes.append(Hitbox(x, y, width, height, self.screen))
        self.hitboxes = new_hitboxes

    # @Override: Handle animation frames instead of singular image
    def reflect_object(self, going_left: bool):
        new_frames = []
        for frame in self.frames:
            new_frames.append(pygame.transform.flip(frame, True, False)) # Flip each frame horizontally
        self.frames = new_frames
        # self.image = self.frames[self.current_frame_index]
        self.going_left = going_left

    def update_pos(self, x: int, y: int):
        self.x = x
        self.y = y

        if not self.going_left:
            self.image_x = (self.x + self.image_width//2) - 10 - CUBE_DIM * 3
            self.image_y = (self.y + self.image_height//2) - 10 - 75- self.delta_y
        else:
            self.image_x = (x + self.image_width//2) - 10 - 25 - self.delta_x
            self.image_y = (y + self.image_height//2) - 10 - 75- self.delta_y#Centered
    
    def construct_hitboxes(self):
        self.hitboxes = [] #martin edited
        if not self.going_left:
            x = self.image_x + CUBE_DIM * 3
        else:
            x = self.image_x

        self.hitboxes.append(Hitbox(x, self.image_y,  # Top-left corner
            self.image_width,  # Width
            self.image_height, self.screen)# Height
        )
        self.reflect_hitboxes()
    
    def process(self):
        super().process()

    def draw_object(self):
        if not self.finished:
            self.screen.blit(self.frames[self.current_frame_index], (self.image_x, self.image_y))
        else:
            #Else remove hitbox instances
            self.hitboxes = []
      

    # TODO: Add specific hammer attack logic

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
    def __init__(self, space: pymunk.Space, x: int, y: int, width_ground: int, collision_type: int = 1):
        self.body = pymunk.Body(x, y, body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, (width_ground, 10))
        self.shape.body.position = (x + width_ground // 2, y)
        self.shape.friction = 0.8
        self.shape.collision_type = 1
        self.shape.filter = pymunk.ShapeFilter(categories=1, mask=2)
        space.add(self.shape, self.body)

class Cube:
    def __init__(self, position: pymunk.Vec2d, space: pymunk.Space, mass: float, collision_type: int, screen: pygame.display, cube_color: tuple[int, int, int, int], state, platforms: list[Ground, Ground], width=50, height=10, health=0):
        self.cool_down = False
        self.cool_down_count = 0
        self.cool_down_duration = -1

        self.mass = mass
        self.width = width
        self.height = height

        # Draw the cube/square shape
        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())
        self.body = pymunk.Body(self.mass, self.moment)
        self.shape.body = self.body

        self.shape.collision_type = collision_type
        self.shape.filter = pymunk.ShapeFilter(categories=2, mask=1)
        self.shape.body.position = position
        space.add(self.shape, self.body)

        # Add eyes using Pymunk circles (to prevent obstruction):
        self.left_eye = pymunk.Circle(self.body, radius=8, offset=(-width // 4, -height // 5))
        self.right_eye = pymunk.Circle(self.body, radius=8, offset=(width // 4, -height // 5))
        self.left_pupil = pymunk.Circle(self.body, radius=4, offset=(-width // 4, -height // 5))
        self.right_pupil = pymunk.Circle(self.body, radius=4, offset=(width // 4, -height // 5))
        self.left_eye.collision_type = collision_type
        self.right_eye.collision_type = collision_type
        self.left_pupil.collision_type = collision_type
        self.right_pupil.collision_type = collision_type
        self.left_eye.filter = pymunk.ShapeFilter(categories=2, mask=1)
        self.right_eye.filter = pymunk.ShapeFilter(categories=2, mask=1)
        self.left_pupil.filter = pymunk.ShapeFilter(categories=2, mask=1)
        self.right_pupil.filter = pymunk.ShapeFilter(categories=2, mask=1)
    
        # Add eyes to the space
        space.add(self.left_eye, self.right_eye, self.left_pupil, self.right_pupil)

        self.shape.color = cube_color
        self.left_eye.color = (255, 255, 255, 255)  
        self.left_pupil.color = (0, 0, 0, 255)
        self.right_eye.color = (255, 255, 255, 255)
        self.right_pupil.color = (0, 0, 0, 255)

        # Health
        self.health = health
        self.lives = 3

        # Movement
        self.direction = 0
        self.move_speed = 50 #On the x
        self.jump_acceleration = 3176

        # Start state = InAirState
        self.state = state

        # Platforms
        self.platforms = platforms

        # GameObjects (weapons & attacks)
        self.attacks: list[Union[GameObject, None]] = [None, None, None]
        self.action = None

        # Store Cube in the body's user_data
        self.body.user_data = self

    def update_cool_down_count(self):
        if(self.cool_down):
            
            if(self.cool_down_count >= self.cool_down_duration):
                self.cool_down = False
                self.cool_down_duration = -1
                self.cool_down_count = 0
            else:
              
                self.cool_down_count += 1
        
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

    def jump(self):
        self.shape.body.apply_force_at_local_point((0, self.jump_acceleration * self.mass * -2 * 1.5), (0, 0))
        self.action = 'jump'
        return self
        
    def update_health(self, new_health: float):
        self.health += int(new_health)

    def get_bounding_box(self) -> pymunk.Shape.cache_bb:
        return self.shape.cache_bb()
    
    def is_on_floor(self) -> bool:
        for platform in self.platforms:
            # Check if object is touching a platform
            if self.shape.cache_bb().intersects(platform.shape.cache_bb()):
                return True
        
   
    def check_current_action(self) -> str:
        return self.action
    """"
    def _physics_process(self, delta: float) -> None:
        new_state: PlayerState = self.state.physics_process(self, delta)

        if new_state != None:
            self.state.exit(self)
            print(self.state.get_state_name(), " -> ", new_state.get_state_name())
            print(self.check_current_action())
            self.state = new_state
            self.state.enter(self)
        
        # if self.state.action == 'light_attack':
        #     side_slash.start_cast()

        # move_and_slide()
"""
    def _physics_process(self, delta: float) -> None:
        self.body.apply_force_at_local_point((0, 20), (0, 0))

        if self.body.velocity.y > 400:  
            self.body.velocity = pymunk.Vec2d(self.body.velocity.x, 400) 

        new_state: PlayerState = self.state.physics_process(self, delta)

        if new_state:
            self.state.exit(self)
            self.state = new_state
            self.state.enter(self)
        
        # Ensure correct state transition when landing
        if self.is_on_floor() and isinstance(self.state, InAirState):
            self.state = GroundState.get_ground_state(self)


# FINITE STATE MACHINE
class State(Enum):
    STANDING = 1
    JUMPING = 2
    ATTACKING = 3

class PlayerState:
    def __init__(self, action = None):
        self.action = action
        self.attacks: dict[str, float] = dict() #{attack, attack_duration(s)}
        self.attacks["sword"] = 500/50
        self.attacks["hammer"] = 50/50
        self.attacks["punch"] = 50/50

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
        elif self.action == 'sword':
            return 'sword'
        elif self.action == 'hammer':
            return 'hammer'
        elif self.action == 'punch':
            return 'punch'
        elif self.action == 'damaged':
            return 'damaged'
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
            
        # Attack:
        if self.check_current_action() == 'damaged': #Check for damaged
            return HurtState().enter(player)
        elif self.check_current_action() is not None and self.check_current_action() in self.attacks.keys(): #Check for attack 
            return AttackingState()
        # Handling jump
        elif self.check_current_action() == 'jump' and player.is_on_floor():
            player.body.velocity = pymunk.Vec2d(player.body.velocity.x, player.jump_speed)
            return InAirState()
        elif not player.is_on_floor(): #Check for air
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

        # print(self.check_current_action())

        # Check for damaged/hurt state:
        if self.check_current_action() == 'damaged':
            return HurtState().enter(player)
        # Attack:
        elif self.check_current_action() is not None and self.check_current_action() in self.attacks.keys(): #Check for attack 
            return AttackingState()
        elif player.is_on_floor():
            return GroundState.get_ground_state(player)
        else:
            return None
            
    def animate_player(self, player):
        pass #Jump should be handled by Pymunk

class HurtState(InAirState):
    def get_state_name(self) -> str:
        return "HurtState"
    
    def enter(self, player: Cube) -> None:
        player.body.velocity = pymunk.Vec2d(player.body.velocity.x, player.body.velocity.y)
        return self

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

class AttackingState(PlayerState):
    def get_state_name(self):
        return "AttackingState"
    
    def physics_process(self, player: Cube, delta: float) -> PlayerState:
        attack_timer: float = 0
        
        if self.check_current_action() is not None:# and player.direction:        
            # Preventing movement during attack
            player.body.velocity = pymunk.Vec2d(0, player.body.velocity.y)

            # Trigger the smash attack when the spacebar is pressed
            hammer = player.attacks[1]
            punch = player.attacks[2]
#martin
         #   hammer.attack.activate()
            punch.attack.activate()
            
            # Check for damaged/hurt state:
            if self.check_current_action() == 'damaged':
                return HurtState().enter(player)
            if player.is_on_floor():
                return GroundState.get_ground_state(player)
            else:
                return InAirState()

        #     if attack_timer >= self.attacks[self.action]:
        #         #Set action back to None when finished & transition to either GroundState or AirState
        #         if player.is_on_floor():
        #             return GroundState.get_ground_state(player)
        #         else:
        #             return InAirState()
        #     else:
        #         attack_timer += DELTA #Add to the timer
        #         self.action = None #????
        #         return None
        # else:
        #     if player.is_on_floor():
        #         return GroundState.get_ground_state(player)
        #     else:
        #         return InAirState()

# Hitboxes
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
        #this one draws the regular one
       
       # self.hitbox_rect = pygame.draw.rect(self.screen, (255, 0, 255), (self.x, self.y, self.width, self.height), 1)
    
    def draw(self):
        pass
        # pygame.draw.rect(self.screen, (255, 0, 0), (self.x, self.y, self.width, self.height), 1)
       # self.hitbox_rect.topleft = (self.x, self.y)
     

class HitboxHandler:
    '''
    This class handles the hitbox interactions between Pygame (GameObject) and Pymunk (agent) objects.
    '''
    def __init__(self, screen: pygame.display):
        self.screen = screen

    def object_hits_agent(self, game_objects: list[GameObject], attacked_agent: Cube, attacking_agent: Cube) -> bool:
        for game_obj in game_objects:
            # Handle NoneType exception
            if game_obj is None:
                continue

            game_obj_hitboxes = game_obj.hitboxes
           
            agent_bb = attacked_agent.get_bounding_box()

            # Bounding box rectangle for agent
       
            pygame.draw.rect(self.screen, (0, 255, 0), (int(agent_bb.left), int(agent_bb.bottom), int(agent_bb.right - agent_bb.left), int(agent_bb.top - agent_bb.bottom)), 1)
            
            for hitbox in game_obj_hitboxes:
                # Draw a rect for the game object
                game_obj_hitbox = pygame.draw.rect(self.screen, (255, 255, 0), (hitbox.x, hitbox.y, hitbox.width, hitbox.height), 1)

                # Check for overlap between a pygame.rect and pymunk bb
                if game_obj_hitbox.colliderect(agent_bb.left, agent_bb.bottom, agent_bb.right - agent_bb.left, agent_bb.top - agent_bb.bottom):

                    attacking_agent.state.action = game_obj.get_object_name().lower() # Update the state's action
                    attacked_agent.update_health(3) # Take damage
                    attacked_agent.state.action = 'damaged'
                    return True 
        
           # attacking_agent.action = None #Refresh the attacking agent's action at the end
        return False
    
    def object_hits_agent_v1(self, game_obj: GameObject, attacked_agent: Cube, attacking_agent: Cube) -> bool:
        game_obj_hitboxes = game_obj.hitboxes
        agent_bb = attacked_agent.get_bounding_box()

        # Bounding box rectangle for agent
        pygame.draw.rect(self.screen, (0, 255, 0), (int(agent_bb.left), int(agent_bb.bottom), int(agent_bb.right - agent_bb.left), int(agent_bb.top - agent_bb.bottom)), 1)
        
        for hitbox in game_obj_hitboxes:
            # Draw a rect for the game object
            game_obj_hitbox = pygame.draw.rect(self.screen, (255, 0, 0), (hitbox.x, hitbox.y, hitbox.width, hitbox.height), 1)

            # Check for overlap between a pygame.rect and pymunk bb
            if game_obj_hitbox.colliderect(agent_bb.left, agent_bb.bottom, agent_bb.right - agent_bb.left, agent_bb.top - agent_bb.bottom):
                # print("hit")
                attacking_agent.state.action = game_obj.get_object_name().lower() # Update the state's action
                attacked_agent.update_health(3) # Take damage
                attacked_agent.state.action = 'damaged'
                return True 
    
        attacking_agent.action = None #Refresh the attacking agent's action at the end
        return False

    # NOTE: Doesn't work
    def apply_hitboxes(self, agent: Cube, game_objects: list[GameObject]):
        
        '''
        Applies hitbox UI and execution logic to the game objects
        '''
        for game_obj in game_objects:
            if game_obj is None:
                continue
            game_obj.update_pos(agent.shape.body.position.x, agent.shape.body.position.y)
            
            if game_obj.attack is not None:
                
                game_obj.attack.execute(game_obj=game_obj, hitbox_handler=self)

            game_obj.update_hitbox_pos(agent.shape.body.position.x, agent.shape.body.position.y)
           
            for hitbox in game_obj.hitboxes:
                hitbox.draw()
            game_obj.draw_object()

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

        # Score images
        SCALE_FACTOR = 0.20
        self.agent_1_score = pygame.image.load('assets/player1score.png')
        self.agent_1_score = pygame.transform.scale(self.agent_1_score, (int(SCALE_FACTOR * self.agent_1_score.get_width()), int(SCALE_FACTOR * self.agent_1_score.get_height())))
        self.agent_2_score = pygame.image.load('assets/player2score.png')
        self.agent_2_score = pygame.transform.scale(self.agent_2_score, (int(SCALE_FACTOR * self.agent_2_score.get_width()), int(SCALE_FACTOR * self.agent_2_score.get_height())))
       
        # Life and death images
        self.life = pygame.image.load('assets/life.png')
        self.life = pygame.transform.scale(self.life, (int(0.075 * self.life.get_width()), int(0.075 * self.life.get_height())))
        self.death = pygame.image.load('assets/death.png')
        self.death = pygame.transform.scale(self.death, (int(0.075 * self.death.get_width()), int(0.075 * self.death.get_height())))

    def display_UI(self):
        # self.display_agent_healths()
        self.display_agent_scores()
        self.display_percentages()

    def display_agent_scores(self):
        self.screen.blit(self.agent_1_score, (25, -35))
        self.screen.blit(self.agent_2_score, (325, -35))

    # @Deprecated: Use custom png icon
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
        font = pygame.font.Font(None, 35)
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
            # text_rect_background = pygame.draw.rect(self.screen, (255,255,255), (220+i*100, 75, 70, 56))
            # text_rect_background_border = pygame.draw.rect(self.screen, (0, 0, 0), (220+i*100, 75, 70, 56), 3)
            text_rect = text_surface.get_rect(center=pygame.Rect(220+i*100, 75, 64, 50).center)
            self.screen.blit(text_surface, text_rect)

            # Agent lives
            for j in range(self.agents[i].lives):
                self.screen.blit(self.life, (50+j*70 + i*250, 162))

            # Agent deaths
            for j in range(3 - self.agents[i].lives):
                self.screen.blit(self.death, (50 + 2*70 - j*70 + i*250, 162))

    def draw_eyes(self, x: float, y: float, radius: int):
        '''
        This function draws eyes
        '''
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        # Draw the white circle (outer part of the eye)
        pygame.draw.circle(self.screen, WHITE, (x, y), radius)
        pygame.draw.circle(self.screen, BLACK, (x, y), radius//2)

class Controller:
    def render_game_objects_and_functions(self, game_objects: list[list[GameObject]], agents: list[Cube], hitbox_handler: HitboxHandler, agent_action_spaces: list[list[int]], frame_count: int):
        # print(game_objects)
        # print(agents)
        # print(agent_action_spaces)
        for a in range(len(agents)): # Loop through each agent
            for i in range(len(game_objects[a])): #Loop through game objects per agent
                # Assuming action spaces look like [1, -1, -1, ..., 1], where -1 represents not doing the action right now, 1 represents doing the action
                if agent_action_spaces[a][i] == 1 and game_objects[a][i] is not None:
        
                    game_objects[a][i].update_pos(agents[0+a].shape.body.position.x, agents[1-a].shape.body.position.y)
  

                    going_left = (agents[1-a].direction == -1)
                    game_objects[a][i].update_pos(agents[a].shape.body.position.x, agents[a].shape.body.position.y)
                    
                    game_objects[a][i].update_hitbox_pos(agents[a].shape.body.position.x, agents[a].shape.body.position.y, going_left)

                    # game_objects[a][i].attack.activate()

                    game_objects[a][i].attack.execute(game_obj=game_objects[a][i], hitbox_handler=hitbox_handler)
                   
                    for hitbox in game_objects[a][i].hitboxes:
                        hitbox.draw()
                    
                    if game_objects[a][i].get_object_name() in ['GroundPound']:
                       game_objects[a][i].process()
                    # game_objects[a][i].process()
                    game_objects[a][i].draw_object()
    
    def check_game_hitboxes(self, agents: list[Cube], hitbox_handler: HitboxHandler):
        for i in range(len(agents)):
            hitbox_handler.object_hits_agent(game_objects=agents[i].attacks, attacking_agent=agents[i], attacked_agent=agents[1-i])

    def process_physics(self, agents: list[Cube], action_spaces: list[list[int]]):
        for i in range(len(agents)):
            for j in range(len(action_spaces[i])):
                agents[i].direction = action_spaces[i][j] #Direction = [-1, 0, 1]
                agents[i]._physics_process(DELTA)
                agents[i].update_cool_down_count()
                break

    def process_lives(self, agents=list[Cube, Cube]):
        for i in range(len(agents)):
            #Reset on death
            if agents[i].shape.body.position.y > 575:
                agents[i].lives -= 1
                agents[i].shape.body.position = pymunk.Vec2d(150 + 300*i, 100)
                agents[i].direction = 1 if i == 0 else -1
                agents[i].health = 0

                if agents[i].lives <= 0:
                    print(f"Agent {i+1} has lost all lives.")
                    sys.exit()


# @Override
"""
def collide(arbiter, space, data):
    # Check for one-way platform collisions
    agents: list[Cube] = [data['agent1'], data['agent2']]
    
    # Check to disable collisions: 
    for agent in agents:
        print(agent.action)
        if agent.body.velocity.y < 0 and agent.state.get_state_name() != 'HurtState':
            agent.shape.filter = pymunk.ShapeFilter(categories=2, mask=0) 
            agent.left_eye = pymunk.ShapeFilter(categories=2, mask=0)
            agent.right_eye = pymunk.ShapeFilter(categories=2, mask=0)
            agent.left_pupil = pymunk.ShapeFilter(categories=2, mask=0)
            agent.right_pupil = pymunk.ShapeFilter(categories=2, mask=0)
            return False
    
    #Enable collisions if not jumping
    for agent in agents:
        agent.shape.filter = pymunk.ShapeFilter(categories=2, mask=1) 
        agent.left_eye = pymunk.ShapeFilter(categories=2, mask=1)
        agent.right_eye = pymunk.ShapeFilter(categories=2, mask=1)
        agent.left_pupil = pymunk.ShapeFilter(categories=2, mask=1)
        agent.right_pupil = pymunk.ShapeFilter(categories=2, mask=1)

    return True
"""

# @Override
def collide(arbiter, space, data):
    agents: list[Cube] = [data['agent1'], data['agent2']]
    
    for agent in agents:
        print(f"Action: {agent.action}")
        if agent.action == 'jump' and agent.body.velocity.y < 0:
            print("Jump detected: Disabling collision.")
            agent.shape.filter = pymunk.ShapeFilter(categories=2, mask=0)
            return False  

        if agent.body.velocity.y >= 0 and agent.is_on_floor():
            print("Landing detected: Enabling collision.")
            agent.shape.filter = pymunk.ShapeFilter(categories=2, mask=1)

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

    # Init 
    state1: PlayerState = InAirState()
    state2: PlayerState = InAirState()

    ball = Cube((150, 100), space, 4, 2, screen, cube_color=(255, 140, 0, 255), state=state1, platforms=[ground, platform1, platform2], width=50, height=50)
    ball2 = Cube((450, 100), space, 3, 2, screen, cube_color=(0, 0, 255, 255), state=state2, platforms=[ground, platform1, platform2], width=50, height=50)

    # Init attacks (GameObjects)
    sword = Sword(150, 100, "./assets/sword.png", screen)
    hammer = Hammer(150, 100, "./assets/hammer.png", screen, ball, ball2)
    punch = Punch(150, 100, "./assets/punch.png", screen, ball, ball2)

    sword2 = Sword(450, 100, "./assets/sword.png", screen)
    hammer2 = Hammer(450, 100, "./assets/hammer.png", screen, ball2, ball)
    punch2 = Punch(450, 100, "./assets/punch.png", screen, ball2, ball)

    gp_test = GroundPound(150, 100, "./assets/unarmedgp.gif", screen, ball, ball2)
    gp_test2 = GroundPound(450, 100, "./assets/unarmedgp.gif", screen, ball2, ball)

    ball.attacks = [sword, hammer, punch, gp_test]
    ball2.attacks = [sword2, hammer2, punch2, gp_test2]
    action_space_1 = [1, 0, -1, -1, -1, 1]
    action_space_2 = [-1, 0, -1, -1, -1, 1]

    hurtbox = Hurtbox()

    # Display initial UI
    ui = UI(screen=screen, agent_1=ball, agent_2=ball2)

    # Game Controller
    controller = Controller()

    hitbox_handler = HitboxHandler(screen)
    collision_handler = space.add_collision_handler(2, 1)
    collision_handler.begin = lambda arbiter, space, data: collide(arbiter, space, {
        'agent1': ball, 
        'agent2': ball2
    })
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

    for _ in range(10000000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    # Trigger the smash attack when the spacebar is pressed
                    # if(hammer.cool_down_count <= 0):
                    hammer.attack.active = True 
                    punch2.attack.activate()
                  

                elif event.key == pygame.K_k:
                    # TEST: manual jump on 'k' pressed
                    ball.jump()

        # hurtbox.hurtbox(ball, ball2)

        # Continuously move the balls toward each other
        # apply_force_toward_each_other(ball, ball2, 50)

        # Watch for FSM and physics_processes
        # print(ball.body.velocity, " ", ball2.body.velocity)
        controller.process_physics(agents=[ball, ball2], action_spaces=[action_space_1[:2], action_space_2[:2]]) # Slicing [:2] gives movement directions
        # ball.direction = 1
        # ball2.direction = -1
        # ball._physics_process(DELTA)
        # ball.update_cool_down_count()
        # ball2._physics_process(DELTA)
        # ball2.update_cool_down_count()
    

        # Quit if any ball goes below the threshold
        # if ball.shape.body.position.y > quit_y_threshold or ball2.shape.body.position.y > quit_y_threshold:
        #     print("A ball has fallen below the threshold. Exiting...")
        #     sys.exit()

        controller.process_lives(agents=[ball, ball2])

        screen.fill((0, 0, 0))

        # Draw GameObject & hammer hitboxes:
        controller.render_game_objects_and_functions(game_objects=[ball.attacks, ball2.attacks], agents=[ball, ball2], hitbox_handler=hitbox_handler, agent_action_spaces=[action_space_1[2:], action_space_2[2:]], frame_count=frame_count) #Slicing [2:] gives attacks
        # hammer.update_pos(ball.shape.body.position.x, ball.shape.body.position.y)
        # hammer.update_hitbox_pos(ball.shape.body.position.x, ball.shape.body.position.y)
        # hammer.attack.execute(game_obj=hammer, hitbox_handler=hitbox_handler)
        
        # punch2.update_pos(ball2.shape.body.position.x, ball2.shape.body.position.y)
        # punch2.update_hitbox_pos(ball2.shape.body.position.x, ball2.shape.body.position.y)
        # punch2.attack.execute(game_obj=punch2, hitbox_handler=hitbox_handler)

        # gp_test.update_pos(ball.shape.body.position.x, ball.shape.body.position.y)
        # gp_test.update_hitbox_pos(ball2.shape.body.position.x, ball2.shape.body.position.y)
        # gp_test.attack.execute(game_obj=gp_test, hitbox_handler=hitbox_handler)

        # frame_idx = frame_count % 26  # Loop through frames
        # gp_test.draw_object(None, frame_idx)
        
        # for hitbox in punch2.hitboxes:
        #     hitbox.draw()
              
        # for hitbox in hammer.hitboxes:
        #     hitbox.draw()
        
        # for hitbox in gp_test.hitboxes:
        #     hitbox.draw()
    
        # hitbox_handler.apply_hitboxes(ball, ball.attacks)
        # hitbox_handler.apply_hitboxes(ball2, ball2.attacks)

        # Check if the GameObject hits Agent for each agent
        controller.check_game_hitboxes(agents=[ball, ball2], hitbox_handler=hitbox_handler)
        # hitbox_handler.object_hits_agent(game_objects=ball.attacks, attacking_agent=ball, attacked_agent=ball2)
        # hitbox_handler.object_hits_agent(game_objects=ball2.attacks, attacking_agent=ball2, attacked_agent=ball)

        # Render health UI
        ui.display_UI()

        # Draw the red line
        pygame.draw.line(screen, (255, 0, 0), (0, quit_y_threshold), (width, quit_y_threshold), 2)

        space.debug_draw(draw_options)
        space.step(DELTA)

        # hammer.draw_object()
        # punch2.draw_object()
        pygame.display.update() # pygame.display.update()
        clock.tick(50)
        # pygame.image.save(screen, f"frames/frame_{frame_count:04d}.png")
        frame_count += 1

if __name__ == '__main__':
    main()