import random
import copy
from game import Game
from constants import Constants
from base_game_model import BaseGameModel

ITER = 300
act = []
iterate = 0
envs_q = []

class MiniMax(BaseGameModel):

    def __init__(self):
        self.long_name = 'Monte Carlo'
    #выбор действия
    def move(self, environment):
        global iterate
        iterate = ITER
        act = self._run2(copy.deepcopy(environment))
        return act

    def find_index(self, arr):
        inde = max(arr)
        count = 0
        for i in arr:
            if i == inde:   count+=1
        
        if count == 1:
            return arr.index(inde)
        else:
            k = random.randint(1, count)
            count = 0
            out = 0
            for i in arr:
                if i == inde:   
                    count+=1
                    if k == count:
                        return out
                out+=1

    #основной алгоритм
    def _run2(self, environment):
        directs = [0, 0, 0]
        acts = environment.possible_actions_for_current_action(environment.snake_action)
        #здесь начинается симуляция во все три стороны 
        for i in range(3):
            score = [0, 0]
            env = copy.deepcopy(environment)
            #в подобных строчках кода происходит оценка текущего состояния и проверяется находится ли "змейка" в терминальном состояние
            st, rew, done, eat = env.full_step_neat(acts[i])
            score = [rew, 1]
            if not(done) and not(eat):   
                #углубление по дереву
                for j in range(10):
                    ENV = copy.deepcopy(env)
                    for k in range(50):
                        
                        st, rew, done, eat = ENV.full_step_neat(random.choice(ENV.possible_actions_for_current_action(ENV.snake_action)))
                        score[0]+=rew
                        score[1]+=1
                        if done or eat:
                            break
        #выбор самого выгодного действия    
            directs[i] = score[0]/score[1]  
        return acts[self.find_index(directs)]

while True:
    Game(game_model=MiniMax(),
        fps=Constants.FPS,
        pixel_size=Constants.PIXEL_SIZE,
        screen_width=Constants.SCREEN_WIDTH,
        screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
        navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)