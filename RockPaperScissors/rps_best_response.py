"""Best-response learning for Rock-Paper-Scissors.

This script learns a strategy against a fixed opponent policy. It keeps track
of regret for each action and updates the player's mixed strategy toward the
best response to the opponent's known strategy.
"""

import random

ROCK, PAPER, SCISSORS, NUM_ACTIONS = 0, 1, 2, 3

# Fixed opponent strategy. The learner is trying to exploit this distribution.
opp_strategy = [0.4, 0.2, 0.4]

# Current mixed strategy and regret totals for the learning player.
p0_strategy = [0.0] * NUM_ACTIONS
p0_strategy_sum = [0.0] * NUM_ACTIONS
p0_regret_sum = [0.0] * NUM_ACTIONS

# Payoff matrix for Rock-Paper-Scissors. Positive values mean the row player wins.
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
    """Update the strategy toward the best response to the fixed opponent."""

    for i in range(iterations):
        # Compute the current mixed strategy from accumulated regrets.
        p0_strategy = getStrategy(p0_regret_sum, p0_strategy_sum)
        p1_strategy = opp_strategy

        # Expected payoff of each action against the fixed opponent.
        p0_utility = [
            sum(p1_strategy[opponent_action] * PAYOFF[player_action][opponent_action]
                for opponent_action in range(NUM_ACTIONS))
            for player_action in range(NUM_ACTIONS)
        ]

        # Expected value of the current mixed strategy.
        p0_expected_utility = sum(
            p0_strategy[action] * p0_utility[action] for action in range(NUM_ACTIONS)
        )

        # Increase regret for actions that performed better than the current mix.
        for action in range(NUM_ACTIONS):
            p0_regret_sum[action] += p0_utility[action] - p0_expected_utility

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