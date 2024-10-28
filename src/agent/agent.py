from typing import List
import torch
import random
import numpy as np
from ..environment.game import SnakeGameAI, Direction, Point
from collections import deque
from ..model import QTrainer, Linear_QNet

MAX_MEMORY = 100_000
BATCH_SIZE = 100_000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        """
        11 states: [danger straight, danger right, danger left, direction left, direction right, direction up,
        direction down, food left, food right, food up, food down
        """
        head = game.head
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        point_mapping: dict[Direction, Point] = {
            Direction.LEFT: Point(head.x - 20, head.y),
            Direction.RIGHT: Point(head.x + 20, head.y),
            Direction.UP: Point(head.x, head.y - 20),
            Direction.DOWN: Point(head.x, head.y + 20),
        }

        state: List[bool] = (
            [
                game.is_collision(point_mapping[game.direction]),
                game.is_collision(
                    point_mapping[
                        clock_wise[(clock_wise.index(game.direction) + 1) % 4]
                    ]
                ),
                game.is_collision(
                    point_mapping[
                        clock_wise[(clock_wise.index(game.direction) - 1) % 4]
                    ]
                ),
            ]
            + [
                True if game.direction == dir else False
                for dir in [
                    Direction.LEFT,
                    Direction.RIGHT,
                    Direction.UP,
                    Direction.DOWN,
                ]
            ]
            + [  # Food location
                game.food.x < game.head.x,  # food left
                game.food.x > game.head.x,  # food right
                game.food.y < game.head.y,  # food up
                game.food.y > game.head.y,  # food down
            ]
        )

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tade off between exploration vs exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move: int = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
