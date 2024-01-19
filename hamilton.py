from point import Point
from node import Node
from base_game_model import BaseGameModel
from longest_path import LongestPathSolver
from game import Game
from constants import Constants


class Hamilton(BaseGameModel):

    hamilton_path = []

    def __init__(self):
        self.long_name = 'Hamilton'

    def move(self, environment):
        BaseGameModel.move(self, environment)
        if environment.is_in_fruitless_cycle():
            print ("Infinite fruitless cycle - game over at: " + str(environment.reward()))
            return environment.snake_action

        hamilton_path = self._hamilton_path(environment)

        for index in range(0, len(hamilton_path)):
            node = hamilton_path[index]
            next_index = index + 1
            if next_index == len(hamilton_path):
                return hamilton_path[0].action
            elif node == self.starting_node:
                return hamilton_path[next_index].action
        return environment.snake_action

    def reset(self):
        self.hamilton_path = []

    def _hamilton_path(self, environment):
        head = self.starting_node
        inverse_snake_action = (environment.snake_action[0] * -1, environment.snake_action[1] * -1)
        tail = environment.snake[-1]
        tail_point = Point(tail.x + inverse_snake_action[0], tail.y + inverse_snake_action[1])
        tail = Node(tail_point)
        if self.hamilton_path:
            return self.hamilton_path
        #выше инициализиуруем стартовую позицию головы и хвоста(так при длине = 1 хвоста нет берется задняя для головы клетка)
        #ищем самый длинный путь от головы до хвоста)
        longest_path_solver = LongestPathSolver()
        self.hamilton_path = longest_path_solver.longest_path(head, tail, environment)
        return self.hamilton_path

while True:
    Game(game_model=Hamilton(),
        fps=Constants.FPS,
        pixel_size=Constants.PIXEL_SIZE,
        screen_width=Constants.SCREEN_WIDTH,
        screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
        navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)