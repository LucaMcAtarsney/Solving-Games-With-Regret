"""Regret-matching for Rock-Paper-Scissors.

This script trains both players in a zero-sum Rock-Paper-Scissors game by
updating regret for each action and converting that regret into a mixed
strategy. Over many iterations, the average strategy approaches the Nash
equilibrium, where each action is played with roughly equal probability.

The average strategy converges to [1/3, 1/3, 1/3].
"""

import random

ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0, 1, 2, 3

# Player 0's current mixed strategy and regret totals.
p0_strategy = [0.0] * NUM_ACTIONS
p0_strategy_sum = [0.0] * NUM_ACTIONS
p0_regret_sum = [0.0] * NUM_ACTIONS

# Player 1's current mixed strategy and regret totals.
p1_strategy = [0.0] * NUM_ACTIONS
p1_strategy_sum = [0.0] * NUM_ACTIONS
p1_regret_sum = [0.0] * NUM_ACTIONS

# Zero-sum payoff matrix for Rock-Paper-Scissors.
# Positive values mean the row player wins.
PAYOFF = (
    (0, -1, 1),
    (1, 0, -1),
    (-1, 1, 0),
)


def getStrategy(regret_sum, strategy_sum):
    """Convert regret values into a mixed strategy.

    If some actions have positive regret, we weight them proportionally. If all
    regrets are zero or negative, we fall back to a uniform strategy.
    """

    strategy = [max(regret, 0.0) for regret in regret_sum]
    normalising_sum = sum(strategy)

    if normalising_sum > 0:
        strategy = [value / normalising_sum for value in strategy]
    else:
        strategy = [1.0 / NUM_ACTIONS] * NUM_ACTIONS

    for action in range(NUM_ACTIONS):
        strategy_sum[action] += strategy[action]

    return strategy


def getAction(strategy):
    """Sample an action from probability distribution."""

    threshold = random.random()
    action = 0
    cumulative_prob = 0.0

    while action < NUM_ACTIONS - 1:
        cumulative_prob += strategy[action]
        if cumulative_prob > threshold:
            break
        action += 1

    return action


def train(iterations):
    """Run regret matching for both players until their average strategies converge."""

    for i in range(iterations):
        # Update each player's mixed strategy from their accumulated regret.
        p0_strategy = getStrategy(p0_regret_sum, p0_strategy_sum)
        p1_strategy = getStrategy(p1_regret_sum, p1_strategy_sum)

        # Expected payoff for each action against the opponent's current strategy.
        p0_utility = [
            sum(p1_strategy[opponent_action] * PAYOFF[player_action][opponent_action]
                for opponent_action in range(NUM_ACTIONS))
            for player_action in range(NUM_ACTIONS)
        ]

        p1_utility = [
            sum(p0_strategy[opponent_action] * PAYOFF[player_action][opponent_action]
                for opponent_action in range(NUM_ACTIONS))
            for player_action in range(NUM_ACTIONS)
        ]

        # Expected value of each player's current mixed strategy.
        p0_expected_utility = sum(
            p0_strategy[action] * p0_utility[action] for action in range(NUM_ACTIONS)
        )

        p1_expected_utility = sum(
            p1_strategy[action] * p1_utility[action] for action in range(NUM_ACTIONS)
        )

        # Update regrets. Positive regret means an action did better than the
        # current mixed strategy and should be weighted more in future rounds.
        for action in range(NUM_ACTIONS):
            p0_regret_sum[action] += p0_utility[action] - p0_expected_utility
            p1_regret_sum[action] += p1_utility[action] - p1_expected_utility

        if i % 1000 == 0:
            print(f"finished iteration {i} ...")


def getAverageStrategy(strategy_sum):
    """Return the average strategy after many training iterations."""

    normalising_sum = sum(strategy_sum)
    if normalising_sum > 0:
        return [value / normalising_sum for value in strategy_sum]
    return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


if __name__ == "__main__":
    train(1_000_000)
    print(getAverageStrategy(p0_strategy_sum))