import pygame
import sys
from pygame.locals import *
from point import Point
from action import Action
from constants import Constants
from environment import Environment
from os.path import join


class Game:

    pygame.init()
    pygame.display.set_caption(Constants.SLITHERIN_NAME)
    icon = pygame.image.load(Constants.ICON_PATH)
    pygame.display.set_icon(icon)
    screen_objects = []
    model = None
    stats = ""

    def __init__(self, game_model, fps, pixel_size, screen_width, screen_height, navigation_bar_height):

        
        self.fps = fps
        self.pixel_size = pixel_size
        self.navigation_bar_height = navigation_bar_height
        self.cell = int(screen_width/pixel_size)
        self.screen = pygame.display.set_mode((screen_width, screen_height), 0, Constants.SCREEN_DEPTH)
        self.surface = pygame.Surface((screen_width, screen_height-navigation_bar_height))
        self.head_up = pygame.image.load('src/Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('src/Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('src/Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('src/Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('src/Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('src/Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('src/Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('src/Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('src/Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('src/Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('src/Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('src/Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('src/Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('src/Graphics/body_bl.png').convert_alpha()

        self.apple = pygame.image.load('src/Graphics/apple.png').convert_alpha()

        self.font = pygame.font.Font(join("src/Font/magic.TTF"), 25)

        self.horizontal_pixels = screen_width / pixel_size+2
        self.vertical_pixels = (screen_height-navigation_bar_height) / pixel_size+2

        self.environment = Environment(width=self.horizontal_pixels,
                                       height=self.vertical_pixels)

        self.screen_width = screen_width


        self.head = None
        self.tail = None

        self.action = Action.right
        if self.cell%2 == 0:
            self.points_fruit = [Point((self.cell-3), (self.cell//2-1))]
        else:
            self.points_fruit = [Point((self.cell-3), ((self.cell-1)//2))]
        if self.cell%2 == 0:
            self.points_snake = [Point(2, (self.cell//2-1))]
        else: 
            self.points_snake = [Point(2, ((self.cell-1)//2))]

        self.model = game_model
        while True:
            self._handle_user_input()
            pygame.time.Clock().tick(fps)
            ai_action = self.model.move(self.environment)
            self.action = ai_action
            self.environment.full_step(ai_action)
            if  self.environment.terminal:
                self.action = Action.right
                self.model.reset()
                self.environment.set_fruit()
                self.environment.set_snake()
            self.points_snake = []
            for i in range(0, len(self.environment.snake)):
                self.points_snake.append(Point(self.environment.snake[i].x-1,self.environment.snake[i].y-1))
            self.points_fruit = []
            self.points_fruit.append(Point(self.environment.fruit[0].x-1,self.environment.fruit[0].y-1))
            self.draw_elements()
            self._display()



    def draw_grass(self):
        grass_color1 = (167,209,61)
        grass_color2 = (175,215,70)
        for row in range(self.cell):
            if row % 2 == 0: 
                for col in range(self.cell):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface, grass_color2, grass_rect)
            else:
                for col in range(self.cell):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface, grass_color2, grass_rect)
        pygame.draw.rect(self.surface, (56,74,12), (0, 0, self.cell * self.pixel_size, self.cell * self.pixel_size), 3)
    

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.points_fruit[0].x*self.pixel_size),int(self.points_fruit[0].y*self.pixel_size), self.pixel_size, self.pixel_size)
        self.surface.blit(self.apple,fruit_rect)

    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()

        for index,point in enumerate(self.points_snake):
            x_pos = point.x * self.pixel_size
            y_pos = point.y * self.pixel_size
            block_rect = pygame.Rect(x_pos,y_pos,self.pixel_size,self.pixel_size)
            if index == 0:
                self.surface.blit(self.head,block_rect)
            elif index == len(self.points_snake) - 1 and len(self.points_snake)>2:
                self.surface.blit(self.tail,block_rect)
            elif len(self.points_snake) > 2:
                previous_block = self.points_snake[index + 1] - point
                next_block = self.points_snake[index - 1] - point
                if previous_block.x == next_block.x:
                    self.surface.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    self.surface.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        self.surface.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        self.surface.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        self.surface.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        self.surface.blit(self.body_br,block_rect)
            else:
                if self.head == self.head_up or self.head == self.head_down:
                    self.surface.blit(self.body_vertical,block_rect)
                else:
                    self.surface.blit(self.body_horizontal,block_rect)

    def update_head_graphics(self):
        if len(self.points_snake) > 1:
            head_relation = self.points_snake[1] - self.points_snake[0]
            if head_relation == Point(1,0): self.head = self.head_left
            elif head_relation == Point(-1,0): self.head = self.head_right
            elif head_relation == Point(0,1): self.head = self.head_up
            elif head_relation == Point(0,-1): self.head = self.head_down
        else:
            if self.action == Action.left: self.head = self.head_left
            elif self.action == Action.right: self.head = self.head_right
            elif self.action == Action.up: self.head = self.head_up
            elif self.action == Action.down: self.head = self.head_down


    def update_tail_graphics(self):
        if len(self.points_snake) > 2:
            tail_relation = self.points_snake[-2] - self.points_snake[-1]
            if tail_relation == Point(1,0): self.tail = self.tail_left
            elif tail_relation == Point(-1,0): self.tail = self.tail_right
            elif tail_relation == Point(0,1): self.tail = self.tail_up
            elif tail_relation == Point(0,-1): self.tail = self.tail_down
        else:
            self.tail_human = None


    def draw_score(self):
        score_text = str(self.environment.reward())
        score_surface = self.font.render(score_text,True,(56,74,12))
        name_surface = self.font.render(self.model.long_name,True,(56,74,12))
        score_x = 40
        score_y = 30
        name_rect = name_surface.get_rect(center = (self.screen_width/2, 20))
        apple_rect = self.apple.get_rect()
        score_rect = score_surface.get_rect(midleft = (40, 20))
        bg_rect = pygame.Rect(0, 0, self.screen_width, 40)

        pygame.draw.rect(self.screen,(167,209,61),bg_rect)
        self.screen.blit(score_surface,score_rect)
        self.screen.blit(self.apple, apple_rect)
        self.screen.blit(name_surface, name_rect)

    def draw_elements(self):
        self.draw_grass()
        self.draw_fruit()
        self.draw_snake()
        self.draw_score()
        self.screen.blit(self.surface, (0, self.navigation_bar_height))

    def _display(self):
        pygame.display.flip()
        pygame.display.update()


    def _handle_user_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                self.model.user_input(event)