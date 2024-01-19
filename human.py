from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from pygame.locals import *
from game import Game

class Human(BaseGameModel):

    action = None

    def __init__(self):
        BaseGameModel.__init__(self, "Human", "human", "hu")

    def move(self, environment):
        BaseGameModel.move(self, environment)
        if self.action is None:
            return environment.snake_action
        backward_action = self.action[0] == environment.snake_action[0] * -1 or self.action[1] == environment.snake_action[1] * -1
        return environment.snake_action if backward_action else self.action

    def user_input(self, event):
        if event.key == K_UP:
            self.action = Action.up
        elif event.key == K_DOWN:
            self.action = Action.down
        elif event.key == K_LEFT:
            self.action = Action.left
        elif event.key == K_RIGHT:
            self.action = Action.right

while True:
            Game(game_model=Human(),
                fps=Constants.FPS,
                pixel_size=Constants.PIXEL_SIZE,
                screen_width=Constants.SCREEN_WIDTH,
                screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)