import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import math
from random import random, sample, choice
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from game import Game
from action import Action


class Memory:
    """
    Класс-буфер для сохранения результатов в формате
    (s, a, r, s', done).
    """
    def __init__(self, capacity):
        """
        :param capacity: размер буфера памяти.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """
        Данный метод сохраняет переданный элемент в циклический буфер.
        :param element: Элемент для сохранения.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.position] = element
            self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        """
        Данный метод возвращает случайную выборку из циклического буфера.
        :param batch_size: Размер выборки.
        :return: Выборка вида [(s1, s2, ... s-i), (a1, a2, ... a-i), (r1, r2, ... r-i),
         (s'1, s'2, ... s'-i), (done1,  done2, ..., done-i)],
            где i = batch_size - 1.
        """
        return list(zip(*sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)
def conv_block(in_channels, out_channels, depth=2, pool = False, drop=False, prob=0.2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    layers.append(nn.ReLU(inplace=True))
    for i in range(depth-1):
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(2))
    if drop:
        layers.append(nn.Dropout2d(p=prob))
    return nn.Sequential(*layers)
class DeepQNetwork(nn.Module):
    """
    Класс полносвязной нейронной сети.
    """
    


    def __init__(self,):
        super().__init__()
        self.conv1 = conv_block(in_channels=4,out_channels=32, pool=True)
        self.conv2 = conv_block(in_channels=32,out_channels=64, pool=True)
        self.fcn = nn.Sequential(nn.AvgPool2d(3), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128,4))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.fcn(x)
        return(x)
    
class Agent:
    """
    Класс агента, обучающегося играть в игру.
    """
    def __init__(self,
                 env,
                 file_name,
                 max_epsilon=1,
                 min_epsilon=0.01,
                 target_update=1024,
                 memory_size=4096,
                 epochs=25,
                 batch_size=64):
        """
        :type env: gym.Env
        :param env gym.Env: Среда, в которой играет агент.
        :param file_name: Имя файла для сохранения и загрузки моделей.
        :param max_epsilon: Макимальная эпсилон для e-greedy police.
        :param min_epsilon: Минимальная эпсилон для e-greedy police.
        :param target_update: Частота копирования параметров из model в target_model.
        :param memory_size: Размер буфера памяти.
        :param epochs: Число эпох обучения.
        :param batch_size: Размер батча.
        """

        self.gamma = 0.97
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        self.file_name = file_name
        self.batch_size = batch_size

        self.device = torch.device("cpu")

        self.memory = Memory(capacity=memory_size)

        self.env = env

        self.model = DeepQNetwork().to(self.device)
        
        self.target_model = DeepQNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()

        self.epochs = epochs

        self.history = []

    def fit(self, batch):
        """
        Метод одной эпохи обучения. Скармливает модели данные,
        считает ошибку, берет градиент и делает шаг градиентного спуска.
        :param batch: Батч данных.
        :return: Возвращает ошибку для вывода в лог.
        """
        state, action, reward, next_state, done = batch

        # Распаковываем батч, оборачиваем данные в тензоры,
        # перемещаем их на GPU

        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            # В этой части кода мы предсказываем максимальное
            # значение q-функции для следующего состояния,
            # см. ур-е Беллмана
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target = reward + self.gamma * q_target

            # Если следующее состояние конечное - то начисляем за него
            # награду за смерть, предусмотренную средой
            #q_target[done] = -7500

        # Предсказываем q-функцию для действий из текущего состояния
        q = self.model(state).gather(1, action.unsqueeze(1))

        # Зануляем градиент, делаем backward, считаем ошибку,
        # делаем шаг оптимизатора
        self.optimizer.zero_grad()

        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()

        for param in self.model.parameters():
            param.data.clamp_(-1, 1)

        self.optimizer.step()
        
        return loss

    def train(self, max_steps=2**10, save_model_freq=100):
        """
        Метод обучения агента.
        :param max_steps: Из-за того, что в некоторых средах
            агент может существовать бесконечно долго,
            необходимо установить максимальное число шагов.
        :param save_model_freq: Частота сохранения параметров модели
        """

        max_steps = max_steps
        loss = 0
        action_all=Action.all()
        for epoch in tqdm(range(self.epochs)):
            step = 0
            done = False

            

            episode_rewards = []

            # Получаем начальное состояние среды
            self.env.set_fruit()
            self.env.set_snake()
            observ = self.env.observation()
            state=self.env.state()
            count = 1
            dist=observ[8]
            score = 0
            reward=0
            fake_reward_pa=0
            count_fl=1
            circle_check = [-1] * 16
            circle_index = 0
            action_66=7
            mindist=observ[8]
            fl=0
            time_out = 0
            hang_out= 0
            hunger = 200
            # Играем одну игру до проигрыша, или пока не сделаем
            # максимальное число шагов
            while not done and hunger!=0:
                step += 1
                time_out += 1
                hunger-=1
                
                # Считаем epsilon для e-greedy police
                epsilon = (self.max_epsilon - self.min_epsilon) * (1 - epoch / self.epochs)
                # if self.env.snake_length < 15:
                #     epsilon = 0.0001
                # else:
                #     epsilon = (self.max_epsilon - self.min_epsilon) * (1 - epoch / self.epochs)
                # epsilon = self.max_epsilon * math.exp(-epoch)
                # Выбираем действие с помощью e-greedy police
                action_choise = self.action_choice(state, epsilon, self.model)
                old_action=action_66
                action_66=action_all[action_choise]
                next_observ, fake_reward_fu, done = self.env.full_step(action_66)
                
                
            
                if(time_out >= math.ceil(count_fl * 0.7 + 10)):
                    reward -= 0.5/count_fl
                    time_out = 0
                if(fake_reward_fu + fake_reward_pa == 0):
                
                    if count_fl==1:
                        size = 2
                    else:
                        size=count_fl
                    reward += math.log(((size) + dist)/((size)+ next_observ[8])) / math.log(size)   
				
                if fake_reward_fu==1 :
                    time_out = 0
                   
                    reward=1
                    hunger=200        
                    count_fl+=1

                if done or hunger==0:
                    reward= -1
			
                if(reward > 1):
                    reward = 1
                elif (reward < -1):
                 reward = -1
                
                episode_rewards.append(reward)
                next_state=self.env.state()
                if done or step == max_steps:
                    # Если игра закончилась, добавляем опыт в память

                    total_reward = sum(episode_rewards)
                    
                    self.memory.push((torch.Tensor(state), action_choise, reward, torch.Tensor(next_state), done))

                    tqdm.write(f'Episode: {epoch},\n' +
                               f'Total reward: {total_reward},\n' +
                               f'Training loss: {loss:.4f},\n' +
                               f'Explore P: {epsilon:.4f},\n' +
                               f'Action: {action_choise}\n' +
                               f'Fruit: {count_fl}\n')

                else:
                    # Иначе - добавляем опыт в память и переходим в новое состояние
                    self.memory.push((torch.Tensor(state), action_choise, reward, torch.Tensor(next_state), done))
                    state = next_state
                    dist=next_observ[8]
                    fake_reward_pa=fake_reward_fu
                    count+=1

            if epoch % self.target_update == 0:
                # Каждые target_update эпох копируем параметры модели в target_model,
                # согласно алгоритму
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % save_model_freq == 0:
                # Каждые save_model_freq эпох сохраняем модель
                # и играем тестовую игру, чтобы оценить модель
                
                self.save_model()

            if epoch > self.batch_size:
                # Поскольку изначально наш буфер пуст, нам нужно наполнить его,
                # прежде чем учить модель. Если буфер достаточно полон, то учим модель.
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))

        self.save_model()
    def action_choice(self, state, epsilon, model):
            
        if random() < epsilon:
            # Выбираем случайное действие из возможных,
                # если случайное число меньше epsilon
            action = choice(torch.arange(4))
        else:
                # Иначе предсказываем полезность каждого действия из даного состояния
            action = model(torch.tensor(torch.Tensor(state).unsqueeze(0)).to(self.device)).view(-1)
                # И берем argmax() от предсказания, чтобы определить, какое действие
                # лучше всего совершить
            
            action = action.max(0)[1].item()

        return action
    def save_model(self, file_name=None):
        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'

        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name=None):
            
        if file_name is None:
            file_name = self.file_name
        path = 'models/' + file_name + '.pth'
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))
class DQN_play(BaseGameModel):
    state_size=19
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self,):
        self.long_name = 'DQN'
        self.model =  Agent(
                                    env=self.prepare_training_environment(),
                                    file_name='snake_7.2',
                                    max_epsilon=1,
                                    min_epsilon=0,
                                    target_update=2000,
                                    epochs=200000,
                                    batch_size=16,
                                    memory_size=8000)
        self.model.load_model()
    

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        state = torch.tensor((environment.state())).float()
        vixod=self.model.action_choice(state=state, epsilon=0, model=self.model.model)
        return self.action_all[vixod]

#print(f"Введите режим 0-если обучение, 1-если игра")
#a = int(input())
"""if a==0:
    model=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    agent = Agent(
    env=model.prepare_training_environment(),
    file_name='snake_7.2',
    max_epsilon=1,
    min_epsilon=0,
    target_update=20,
    epochs=500000,
    batch_size=16,
    memory_size=8000)
    agent.load_model()
    agent.train()
if a==1:"""
while True:
            Game(game_model=DQN_play(),
                fps=Constants.FPS,
                pixel_size=Constants.PIXEL_SIZE,
                screen_width=Constants.SCREEN_WIDTH,
                screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
    