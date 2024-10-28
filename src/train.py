from .agent.agent import Agent
from .environment.game import SnakeGameAI
from .helper import plot


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        current_state = agent.get_state(game)

        # get move
        next_action = agent.get_action(current_state)

        # perform move and get new state and reward

        game_over, score, reward = game.play_step(next_action)
        next_state = agent.get_state(game)

        # train short memory

        agent.train_short_memory(
            state=current_state,
            action=next_action,
            reward=reward,
            next_state=next_state,
            done=game_over,
        )
        agent.remember(
            state=current_state,
            action=next_action,
            reward=reward,
            next_state=next_state,
            done=game_over,
        )

        if game_over:
            # train the long memory
            # trains on all the games it has played
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_score.append(score)
            total_score += score
            mean_scores = total_score / agent.n_games
            plot_mean_scores.append(mean_scores)
            plot(plot_score, plot_mean_scores)
