
import random 
from collections import namedtuple, deque
from base_game_model import BaseGameModel
from action import Action
from pygame.locals import *
import random
from constants import Constants
from game import Game
import copy
from environment import Environment
import math
from action import Action
import numpy as np
import os
import sys
import pickle
from matplotlib import pyplot as plt
import neat 
from math import log
ind = 0
max_fitness = 0
gen = 0
max_score = 0
generation_number = 0
best_foods = 0
best_fitness = 0
loop_punishment = 0.25
near_food_score = 0.2
far_food_score = 0.225
moved_score = 0.01
list_best_fitness = []
fig = plt.figure()
b=0
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename='trained/best_generation_instances.pickle'):
    obj = 0
    if os.path.getsize(filename)>0:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    return obj

# def load_object(filename='neat-checkpoint-9999'):
    # po = neat.Checkpointer.restore_checkpoint("neat-checkpoint-9999")


#Текущий рабочий каталог
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.ini')
#Загрузка конфига
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

#создание изначальной популяции
pop = neat.Population(config)
# pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-377')
#Добавление репортера для отоброжения прогресса
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(10000))



plt.ion()
fig = plt.figure()
plt.title('Best fitness')
ax = fig.add_subplot(111)
line_best_fitness, = ax.plot(list_best_fitness, 'r-')

def save_best_generation_instance(instance, filename='trained/best_generation_instances.pickle'):
    # instances = []
    # if os.path.isfile(filename):
    #     instances = load_object(filename)
    # instances.append(instance)
    save_object(instance, filename)


def eval_fitness(genomes, config):
    """
    Оценивает пригодность предоставленного генома.
    Аргументы:
    genomes: Геном для оценки.
    config: конфигурация гиперпараметров.
    Возвращается:
    Оценка пригодности генотипа
    """
    global best_foods
    
    global best_fitness 
    global loop_punishment 
    global near_food_score 
    global far_food_score 
    global moved_score
    global line_best_fitness
    global b
    global  gen,  max_fitness, ind, max_score
    
    if b!=0:
        state = b.observation2()
    best_instance = None
    genome_number = 0
    gen += 1
    global generation_number
    global pop
    action=Action.all()
    
    nets = []
    snakes = []
    ge = []

    max_fitness = 0
    o = 0
    for genome_id, genome in genomes:
        try:
            if genome.fitness > max_fitness:
                max_fitness = genome.fitness
                ind = o
        except:
            pass
        o += 1
        b.set_snake()
        b.set_fruit()
        genome.fitness = 0  # все геномы начинаются с 0 фитнесом
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(b)
        ge.append(genome)
    
   
    step_score = 1
    food_score = 0.0
    score=0
    hunger = [200]*len(snakes)
    fake_reward_pa=0
    countFrames = 0
    foods = 0
    time_out = [0] * len(snakes)
    hang_out = 0
    count_fl = [1] * len(snakes)
    reward = 0
    outputs = None
    loop = 0
    while len(snakes) > 0:
        countFrames += 1
        for x, snk in enumerate(snakes):

            outputs = nets[x].activate((snakes[x].state2()))
            direction = outputs.index(max(outputs))
            action_66=action[direction]
            hunger[x] -= 1
            time_out[x] += 1
            next_state, fake_reward_fu, done, feat = snakes[x].full_step_neat(action_66)
            if(time_out[x] >= math.ceil(count_fl[x] * 0.7 + 10)):
                ge[x].fitness -= 0.5/count_fl[x]
                time_out[x] = 0
            if(fake_reward_fu  == 0):
                
                    if count_fl[x]==1:
                        size = 2
                    else:
                        size=count_fl[x]
                    
                    ge[x].fitness += math.log(((size) + snakes[x].observation2()[8])/((size)+ next_state[8])) / math.log(size)
            if fake_reward_fu==1 :
                time_out[x] = 0

                ge[x].fitness +=1
                hunger[x]=200        
                count_fl[x]+=1
            if done or hunger[x]==0:
                ge[x].fitness -= 1
                nets.pop(x)
                ge.pop(x)
                snakes.pop(x)
                count_fl.pop(x)
                time_out.pop(x)
                hunger.pop(x)
        

if generation_number % 200 == 0:
    
    save_object(pop, 'trained/population.dat')
    print("Exporting population")
    
list_best_fitness.append(best_fitness)
line_best_fitness.set_ydata(np.array(list_best_fitness))
line_best_fitness.set_xdata(list(range(len(list_best_fitness))))
plt.xlim(0, len(list_best_fitness)-1)
plt.ylim(0, max(list_best_fitness)+0.5)
fig.canvas.draw()
fig.canvas.flush_events()

def eval_genomes(genomes, config):
    """
    Функция для оценки пригодности каждого генома в
    списке геномов.
    Аргументы:
    геномы: Список геномов из популяции в
    текущем поколении
    конфигурация: Настройки конфигурации с алгоритмом
    гиперпараметры
    """
    global best_foods
    best_foods = 0
    for genome_id, genome in genomes:
        
        genome.fitness = eval_fitness( genome, config)
    print("111111111111111111111111111111111111111")
    print(best_foods)
    print("111111111111111111111111111111111111111")



class NEAT_trainer(BaseGameModel):
    global pop
    
    global list_best_fitness
    def __init__(self):
        self.long_name = 'NEAT'
    def move(self, env):
        BaseGameModel.move(self, env)
        global b 
        global pop
        global list_best_fitness
        list_best_fitness = []
         
        b=env
        
        best_evolution = pop.run(eval_fitness, 1000)
        

        

class NEAT_play(BaseGameModel):
    state_size=10
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self,):
        self.long_name = 'NEAT'
        
        g = load_object()
        
        self.net = neat.nn.FeedForwardNetwork.create(g['genome'], config)
        

    def move(self, environment):
        BaseGameModel.move(self, environment)
        outputs = self.net.activate(environment.observation2())
        print(outputs)
        direction = outputs.index(max(outputs))
        print(direction)
        return Action.all()[direction]


print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    
    model=NEAT_trainer()
    model.move(model.prepare_training_environment())
  
if a==1:
    while True:
                Game(game_model=NEAT_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
