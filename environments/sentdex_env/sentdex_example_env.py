import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


'''
An environment where the player aims to reach the food and avoid an enemy

Built from sentdex's YouTube tutorial: 
    https://www.youtube.com/watch?v=G92TF4xYQcU&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=5
'''


################################## Hyperparameters ##################################

start_q_table = "qtable-1758380386.pickle" # or filename

SIZE = 10 # 10 x 10 grid for state space
HM_EPISODES = 25000 # num of episodes
MOVE_PENALTY = 1 
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.0 if start_q_table else 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1 if start_q_table else 3000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1 
FOOD_N = 2
ENEMY_N = 3
d = {1: (255, 175, 0), # mix of blue and green
     2: (0, 255, 0), # very green
     3: (0, 0, 255) # very red
    }


################################## Create class for the blobs ##################################


class Blob:
    def __init__(self):
        """Class for the blobs on the grid
        
        Attributes:
            x: horizontal position
            y: vertical position
        """

        # issue: food, player or enemy spawning on top of eachother
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        """Subtraction operator overide -- under subtraction of Blob objects"""
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        """Performs the action taken by the agent"""
        # can only move diagonally
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        """Performs the movement of a blob"""
        if not x:
            self.x += np.random.randint(-1, 2) # -1, 0 or 1
        else: 
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
            
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


################################## Train agent on the environment ####################################


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    # each observation space needs 4 random values (4 discrete actions)
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)] 

else: 
    # if we have a pretrained qtable
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)    


episode_rewards = []
for episode in range(HM_EPISODES):
    # initialise player, food and enemy
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200): # num time steps
        obs = (player-food, player-enemy) # observation

        # q learning
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else: 
            action = np.random.randint(0, 4) # action 0, 1, 2, or 3

        # take an action
        player.action(action)

        ### ADD ENEMEY AND FOOD MOVEMENT (but not for initial training)
        if start_q_table:
            enemy.move()
            food.move()
        ###

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        # we've moved so make new observation
        new_obs = (player-food, player-enemy)

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        # update q value
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)
        
        # update q table with this new q value
        q_table[obs][action] = new_q

        if show:
            env = np.zeros(((SIZE, SIZE, 3)), dtype = np.uint8) # initialise all black grid
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("Title", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: 
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else: 
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


################################## Define moving average and plot ##################################


# make a moving average
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

# plot moving average vs episode
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel(f"episode #")
plt.show()

# dump qtable
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)




