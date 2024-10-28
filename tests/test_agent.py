from reinforcement_learning.src.agent.agent import Agent, SnakeGameAI, Direction, Point
import numpy as np


def get_state(game):
    head = game.head
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r))
        or (dir_l and game.is_collision(point_l))
        or (dir_u and game.is_collision(point_u))
        or (dir_d and game.is_collision(point_d)),
        # Danger right
        (dir_u and game.is_collision(point_r))
        or (dir_d and game.is_collision(point_l))
        or (dir_l and game.is_collision(point_u))
        or (dir_r and game.is_collision(point_d)),
        # Danger left
        (dir_d and game.is_collision(point_r))
        or (dir_u and game.is_collision(point_l))
        or (dir_r and game.is_collision(point_u))
        or (dir_l and game.is_collision(point_d)),
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y,  # food down
    ]

    return np.array(state, dtype=int)


def test_state(mocker):
    agent = Agent()
    game = SnakeGameAI()
    mocker.patch.object(game, "head", Point(100, 100))
    mocker.patch.object(game, "food", Point(100, 140))
    mocker.patch.object(game, "direction", Direction.RIGHT)

    agent.get_state(game) == get_state(game)
