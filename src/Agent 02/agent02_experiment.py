"""
AGENT02: Reinforcement-Learning-Based Agent vs Rule-Based Agent
================================================================

Course: Intelligent System Development
Student: Phan Van Khoi - B23DCCN447 - D23CTPM01-B

Run:
    python agent02_experiment.py

Dependencies:
    numpy
    matplotlib

Agents:
    1. RL-based agent: tabular Q-learning (reference state (d, c))
    2. Rule-based agent: fixed if-else policy
    3. Extended-state Q-learning agent (d, c, own previous action)
    4. Model-based agent with a learned market/competitor model and
       value-iteration planning, proposed in Question 10

Changes relative to the reference implementation:
    M1. The environment uses one random generator for demand and one
        generator per agent for sales. With the same seed, every
        evaluated pair sees the same demand sequence (common random
        numbers), so RL and rule-based agents are compared fairly.
    M2. All agents share one interface (observe / act / learn), so the
        same training and evaluation loops serve RL-vs-rule, RL-vs-RL
        and the extended-state agent.
    M3. exact_evaluation() computes the expected reward of greedy
        policies by propagating the joint state distribution. It gives
        a noise-free benchmark and the best policy for each state
        representation (best_response).
    M4. Additional experiments: windowed learning curves, fixed epsilon,
        10 independent seeds, two simultaneously learning agents,
        discount-rate and reward-function changes, extended-state agent.

Evaluation protocol (all experiments):
    - Q-tables are frozen and exploration is disabled (epsilon = 0).
    - 500 evaluation episodes x 20 steps = 10,000 steps.
    - Evaluation seed = training seed + 500,000 (baseline) or
      training seed + 800,000 (multi-seed experiments).
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Config:
    episodes: int = 5000
    steps_per_episode: int = 20

    alpha: float = 0.10
    gamma: float = 0.90

    epsilon_start: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995

    price: float = 10.0
    unit_cost: float = 5.0
    discount_rate: float = 0.20

    low_demand_customers: int = 10
    high_demand_customers: int = 20

    # "profit" (reference) or "revenue" (Question 5 / Question 9).
    reward_mode: str = "profit"

    seed: int = 42
    evaluation_episodes: int = 500

    # Independent training seeds for the repeated experiments.
    seeds: tuple = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)

    # K episodes per window for learning curves.
    window: int = 100

    # Exact policy value is recorded every N training episodes.
    track_every: int = 25

    output_dir: str = "agent02_results"


AGENT_NAMES = ["Agent 1", "Agent 2"]


# ============================================================
# 2. MARKET MODEL
# ============================================================

def conversion_probability(demand, own_action, competitor_action):
    """p_i(t) = clip(0.30 + 0.25 d + 0.15 a_i - 0.10 a_-i, 0.05, 0.95)."""
    p = (
        0.30
        + 0.25 * demand
        + 0.15 * own_action
        - 0.10 * competitor_action
    )
    return min(max(p, 0.05), 0.95)


def effective_price(config, action):
    return config.price * (1.0 - config.discount_rate * action)


def profit_per_sale(config, action):
    return effective_price(config, action) - config.unit_cost


def reward_per_sale(config, action):
    if config.reward_mode == "revenue":
        return effective_price(config, action)
    return profit_per_sale(config, action)


def customers_for(config, demand):
    if demand == 0:
        return config.low_demand_customers
    return config.high_demand_customers


# ============================================================
# 3. E-COMMERCE ENVIRONMENT
# ============================================================

class ECommerceEnvironment:
    """
    Two agents compete in a simplified e-commerce market.

    Joint state stored by the environment:
        demand d_t in {0, 1}
        previous_actions = [a_1(t-1), a_2(t-1)]

    Each agent builds its own observation from this joint state
    through agent.observe(). The reference observation is
        s_i(t) = (d_t, a_-i(t-1)).
    """

    def __init__(self, config, seed=None):
        self.config = config
        actual_seed = config.seed if seed is None else seed

        # M1: separate streams for demand and for each agent's sales.
        streams = np.random.SeedSequence(actual_seed).spawn(3)
        self.demand_rng = np.random.default_rng(streams[0])
        self.sales_rngs = [
            np.random.default_rng(stream) for stream in streams[1:]
        ]

        self.demand = 0
        self.previous_actions = (0, 0)

    def reset(self):
        self.demand = int(self.demand_rng.integers(0, 2))
        self.previous_actions = (0, 0)

    def step(self, actions):
        if len(actions) != 2:
            raise ValueError("Exactly two actions are required.")

        if any(action not in (0, 1) for action in actions):
            raise ValueError("Actions must be 0 or 1.")

        customers = customers_for(self.config, self.demand)

        rewards = []
        profits = []
        sales_list = []

        # Both actions are selected before this function is called.
        # Neither agent observes the other's current action beforehand.
        for i in range(2):
            conversion = conversion_probability(
                self.demand, actions[i], actions[1 - i]
            )

            sales = int(self.sales_rngs[i].binomial(customers, conversion))

            rewards.append(sales * reward_per_sale(self.config, actions[i]))
            profits.append(sales * profit_per_sale(self.config, actions[i]))
            sales_list.append(sales)

        # Store current actions for the next observation.
        self.previous_actions = tuple(actions)

        # Demand evolves independently of current actions.
        self.demand = int(self.demand_rng.integers(0, 2))

        info = {
            "sales": sales_list,
            "profits": profits,
            "customers": customers
        }

        return rewards, info


# ============================================================
# 4. REINFORCEMENT-LEARNING-BASED AGENTS
# ============================================================

class QLearningAgent:
    """
    Tabular Q-learning agent with the reference state.

    Q[demand, competitor_previous_action, action]
    Total Q-values = 2 * 2 * 2 = 8
    """

    learns = True
    state_shape = (2, 2)
    state_labels = ("d", "c")

    def __init__(
        self,
        alpha=0.10,
        gamma=0.90,
        epsilon_start=0.20,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        seed=123
    ):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

        self.rng = np.random.default_rng(seed)

        self.q = np.zeros(self.state_shape + (2,), dtype=float)

    @classmethod
    def from_config(cls, config, seed, **overrides):
        params = {
            "alpha": config.alpha,
            "gamma": config.gamma,
            "epsilon_start": config.epsilon_start,
            "epsilon_min": config.epsilon_min,
            "epsilon_decay": config.epsilon_decay,
            "seed": seed
        }
        params.update(overrides)
        return cls(**params)

    def observe(self, demand, previous_actions, i):
        return (demand, previous_actions[1 - i])

    def set_episode(self, episode):
        # Decrease exploration as training progresses.
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * self.epsilon_decay ** episode
        )

    def finalize(self):
        pass

    def act(self, state, explore=True):
        # Exploration.
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))

        # Exploitation with random tie-breaking.
        q0, q1 = self.q[state]

        if math.isclose(q0, q1, rel_tol=1e-9, abs_tol=1e-12):
            return int(self.rng.integers(0, 2))

        return int(q1 > q0)

    def discount_probability(self, state):
        """Probability of action 1 under the greedy evaluation policy."""
        q0, q1 = self.q[state]

        if math.isclose(q0, q1, rel_tol=1e-9, abs_tol=1e-12):
            return 0.5

        return float(q1 > q0)

    def learn(self, state, action, reward, next_state, terminal=False):
        old_value = self.q[state + (action,)]

        if terminal:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])

        # Q-learning update.
        self.q[state + (action,)] = old_value + self.alpha * (
            target - old_value
        )

    def states(self):
        return list(np.ndindex(self.state_shape))

    def greedy_signature(self):
        """Greedy action per state; -1 marks a tie."""
        return tuple(
            -1 if self.discount_probability(s) == 0.5
            else int(self.discount_probability(s))
            for s in self.states()
        )


class ExtendedQLearningAgent(QLearningAgent):
    """
    Question 10: Q-learning with the agent's own previous action.

    s_i(t) = (d_t, a_-i(t-1), a_i(t-1))

    The rule-based competitor reacts to the agent's previous action,
    so this extra variable makes the observation Markov for the
    RL-vs-rule game. Total Q-values = 2 * 2 * 2 * 2 = 16.
    """

    state_shape = (2, 2, 2)
    state_labels = ("d", "c", "own")

    def observe(self, demand, previous_actions, i):
        return (demand, previous_actions[1 - i], previous_actions[i])


class ModelBasedAgent(ExtendedQLearningAgent):
    """
    Question 10: proposed agent.

    State: (d, c, own previous action), as in ExtendedQLearningAgent.
    Learning: instead of a single noisy TD update, the agent counts
        N(s, a), the reward sum and next-state frequencies, giving
        R_hat(s, a) and P_hat(s' | s, a). P_hat contains the demand
        process and a model of the competitor's reaction.
    Planning: value iteration on the learned model every 50 episodes
        and once after training. plan_gamma = 0.98 is closer to the
        average-reward objective used in evaluation.
    Exploration: same epsilon-greedy schedule as the reference agent.
    """

    plan_gamma = 0.98
    plan_every = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_states = int(np.prod(self.state_shape))
        self.visits = np.zeros((n_states, 2))
        self.reward_sum = np.zeros((n_states, 2))
        self.transitions = np.zeros((n_states, 2, n_states))

    def index(self, state):
        return int(np.ravel_multi_index(state, self.state_shape))

    def learn(self, state, action, reward, next_state, terminal=False):
        i = self.index(state)
        self.visits[i, action] += 1
        self.reward_sum[i, action] += reward
        self.transitions[i, action, self.index(next_state)] += 1

    def set_episode(self, episode):
        super().set_episode(episode)

        if episode % self.plan_every == 0:
            self.plan()

    def finalize(self):
        self.plan()

    def plan(self):
        visits = np.maximum(self.visits, 1)
        reward = self.reward_sum / visits
        transition = self.transitions / visits[..., None]

        value = np.zeros(reward.shape[0])

        for _ in range(1000):
            q = reward + self.plan_gamma * transition @ value
            new_value = q.max(axis=1)

            if np.max(np.abs(new_value - value)) < 1e-8:
                break

            value = new_value

        self.q = q.reshape(self.state_shape + (2,))


# ============================================================
# 5. RULE-BASED AND FIXED-POLICY AGENTS
# ============================================================

class RuleBasedAgent:
    """
    Fixed policy:

    Rule 1: Low demand -> no discount.
    Rule 2: High demand and competitor previously discounted
            -> offer a discount.
    Rule 3: High demand and competitor did not previously discount
            -> no discount.

    This agent does not learn from rewards.
    """

    learns = False

    def observe(self, demand, previous_actions, i):
        return (demand, previous_actions[1 - i])

    def set_episode(self, episode):
        pass

    def finalize(self):
        pass

    def act(self, state, explore=False):
        demand, competitor_previous_action = state

        if demand == 0:
            return 0

        elif demand == 1 and competitor_previous_action == 1:
            return 1

        else:
            return 0

    def discount_probability(self, state):
        return float(self.act(state))


class TitForTatAgent(RuleBasedAgent):
    """
    Alternative competitor used in the robustness test (Question 10):
    copy the competitor's previous action, regardless of demand.
    """

    def act(self, state, explore=False):
        return int(state[1])


class TablePolicyAgent(RuleBasedAgent):
    """Deterministic lookup-table policy over a chosen observation."""

    def __init__(self, table, observe_fn):
        self.table = table
        self.observe_fn = observe_fn

    def observe(self, demand, previous_actions, i):
        return self.observe_fn(demand, previous_actions, i)

    def act(self, state, explore=False):
        return self.table[state]


# ============================================================
# 6. EXACT EXPECTED REWARD OF FROZEN POLICIES
# ============================================================

def exact_evaluation(config, agents):
    """
    Expected mean reward/profit per step and discount rate of two
    frozen greedy policies under the evaluation protocol:
    T-step episodes, uniform initial demand, previous actions (0, 0).

    The joint state (d, a_1(t-1), a_2(t-1)) is Markov, so its
    distribution is propagated step by step without sampling.
    """

    T = config.steps_per_episode

    distribution = {(0, 0, 0): 0.5, (1, 0, 0): 0.5}

    reward = np.zeros(2)
    profit = np.zeros(2)
    discount = np.zeros(2)

    for _ in range(T):
        next_distribution = defaultdict(float)

        for (demand, p0, p1), mass in distribution.items():
            previous = (p0, p1)
            customers = customers_for(config, demand)

            probs = [
                agent.discount_probability(
                    agent.observe(demand, previous, i)
                )
                for i, agent in enumerate(agents)
            ]

            for actions in product((0, 1), repeat=2):
                weight = mass
                for i in range(2):
                    weight *= probs[i] if actions[i] else 1.0 - probs[i]

                if weight == 0.0:
                    continue

                for i in range(2):
                    sales = customers * conversion_probability(
                        demand, actions[i], actions[1 - i]
                    )
                    reward[i] += weight * sales * reward_per_sale(
                        config, actions[i]
                    )
                    profit[i] += weight * sales * profit_per_sale(
                        config, actions[i]
                    )
                    discount[i] += weight * actions[i]

                for next_demand in (0, 1):
                    next_distribution[
                        (next_demand,) + actions
                    ] += 0.5 * weight

        distribution = next_distribution

    return {
        "reward": reward / T,
        "profit": profit / T,
        "discount_rate": discount / T
    }


def best_response(config, opponent, agent_class=QLearningAgent):
    """
    Best deterministic stationary policy for seat 0 over the state
    representation of agent_class, found by exhaustive enumeration
    (16 policies for the reference state, 256 for the extended one).
    """

    prototype = agent_class()
    states = prototype.states()

    best_value = -np.inf
    best_table = None

    for bits in product((0, 1), repeat=len(states)):
        table = dict(zip(states, bits))
        policy = TablePolicyAgent(table, prototype.observe)

        value = exact_evaluation(config, [policy, opponent])["reward"][0]

        if value > best_value + 1e-9:
            best_value = value
            best_table = table

    return best_table, best_value


# ============================================================
# 7. TRAINING
# ============================================================

def run_training(config, agents, seed, opponent_for_tracking=None):
    """
    Train the learning agents in `agents` (fixed agents stay fixed).

    Returns a dictionary with per-episode rows, per-episode Q-tables
    and greedy signatures of agent 0, and the exact value of agent 0's
    greedy policy every config.track_every episodes.
    """

    env = ECommerceEnvironment(config, seed=seed)
    T = config.steps_per_episode

    rows = []
    q_history = []
    signatures = [[] for _ in agents]
    tracked = []

    for episode in range(config.episodes):

        for agent in agents:
            agent.set_episode(episode)

        env.reset()

        episode_rewards = [0.0, 0.0]
        episode_discounts = [0, 0]

        for step in range(T):

            # Both agents choose actions before the environment changes.
            states = [
                agent.observe(env.demand, env.previous_actions, i)
                for i, agent in enumerate(agents)
            ]

            actions = [
                agent.act(states[i], explore=True)
                for i, agent in enumerate(agents)
            ]

            rewards, info = env.step(actions)

            terminal = step == T - 1

            for i, agent in enumerate(agents):
                if agent.learns:
                    next_state = agent.observe(
                        env.demand, env.previous_actions, i
                    )
                    agent.learn(
                        states[i], actions[i], rewards[i],
                        next_state, terminal
                    )

                episode_rewards[i] += rewards[i]
                episode_discounts[i] += actions[i]

        row = {"episode": episode + 1}

        for i, agent in enumerate(agents):
            row[f"agent{i + 1}_episode_reward"] = episode_rewards[i]
            row[f"agent{i + 1}_mean_reward_per_step"] = episode_rewards[i] / T
            row[f"agent{i + 1}_discount_rate"] = episode_discounts[i] / T
            row[f"agent{i + 1}_epsilon"] = getattr(agent, "epsilon", 0.0)

            if agent.learns:
                signatures[i].append(agent.greedy_signature())

        rows.append(row)

        if agents[0].learns:
            q_history.append(agents[0].q.ravel().copy())

        if (episode + 1) % config.track_every == 0:
            opponents = agents if opponent_for_tracking is None else [
                agents[0], opponent_for_tracking
            ]
            tracked.append({
                "episode": episode + 1,
                "greedy_value": exact_evaluation(
                    config, opponents
                )["reward"][0]
            })

    for agent in agents:
        agent.finalize()

    return {
        "rows": rows,
        "q_history": np.array(q_history),
        "signatures": signatures,
        "tracked": tracked
    }


def train_rl_against_rule(config, seed, agent_class=QLearningAgent,
                          opponent=None, **agent_overrides):
    """Train one RL agent against a fixed rule-based opponent."""

    rl_agent = agent_class.from_config(
        config, seed=seed + 10_000, **agent_overrides
    )
    opponent = RuleBasedAgent() if opponent is None else opponent

    result = run_training(config, [rl_agent, opponent], seed)

    return rl_agent, result


# ============================================================
# 8. SEPARATE POLICY EVALUATION
# ============================================================

def evaluate_pair(config, agents, episodes=None, seed=900_000):
    """
    Evaluate two frozen policies.

    Important:
        - No Q-table updates occur.
        - RL exploration is disabled.
        - Evaluation uses a separate environment random seed.
    """

    if episodes is None:
        episodes = config.evaluation_episodes

    env = ECommerceEnvironment(config, seed=seed)
    T = config.steps_per_episode

    total_rewards = np.zeros(2)
    total_profits = np.zeros(2)
    action_counts = np.zeros((2, 2), dtype=int)

    episode_results = []

    for episode in range(episodes):

        env.reset()
        rewards_this_episode = [0.0, 0.0]

        for _ in range(T):
            states = [
                agent.observe(env.demand, env.previous_actions, i)
                for i, agent in enumerate(agents)
            ]

            # Greedy policy: no exploration.
            actions = [
                agent.act(states[i], explore=False)
                for i, agent in enumerate(agents)
            ]

            rewards, info = env.step(actions)

            for i in range(2):
                total_rewards[i] += rewards[i]
                total_profits[i] += info["profits"][i]
                rewards_this_episode[i] += rewards[i]
                action_counts[i, actions[i]] += 1

        episode_results.append({
            "episode": episode + 1,
            "agent1_reward": rewards_this_episode[0],
            "agent2_reward": rewards_this_episode[1],
            "agent1_mean_reward_per_step": rewards_this_episode[0] / T,
            "agent2_mean_reward_per_step": rewards_this_episode[1] / T
        })

    total_steps = episodes * T

    summary = []

    for i in range(2):
        summary.append({
            "seat": AGENT_NAMES[i],
            "total_reward": total_rewards[i],
            "mean_reward_per_step": total_rewards[i] / total_steps,
            "mean_profit_per_step": total_profits[i] / total_steps,
            "no_discount_actions": int(action_counts[i, 0]),
            "discount_actions": int(action_counts[i, 1]),
            "discount_action_rate": action_counts[i, 1] / total_steps
        })

    return summary, episode_results


# ============================================================
# 9. METRICS, CSV AND Q-TABLE OUTPUT
# ============================================================

def window_means(values, window):
    """Mean reward per step over consecutive windows of K episodes."""
    values = np.asarray(values, dtype=float)
    usable = len(values) // window * window
    return values[:usable].reshape(-1, window).mean(axis=1)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std


def policy_changes(signatures, start):
    """Number of greedy-policy changes after episode `start`."""
    return sum(
        signatures[k] != signatures[k - 1]
        for k in range(max(start, 1), len(signatures))
    )


def write_csv(filepath, rows):
    if not rows:
        return

    # Rows of different configurations may carry different columns.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def format_policy(agent):
    names = {0: "NO DISCOUNT", 1: "DISCOUNT", -1: "TIE"}
    lines = []

    for state, action in zip(agent.states(), agent.greedy_signature()):
        label = ", ".join(
            f"{name}={value}"
            for name, value in zip(agent.state_labels, state)
        )
        lines.append(f"State ({label}): {names[action]}")

    return lines


def save_q_table(filepath, agent, title):
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(
            f"Q[{', '.join(agent.state_labels)}, action]; "
            "Action 0 = no discount; Action 1 = discount\n"
        )

        for state, action in zip(agent.states(), agent.greedy_signature()):
            q0, q1 = agent.q[state]
            file.write(
                f"State={state} | Q0={q0:.6f} | Q1={q1:.6f} | "
                f"Greedy action={action}\n"
            )

        file.write("\n")


# ============================================================
# 10. PLOTS
# ============================================================

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def save_figure(filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def plot_training_rewards(rows, config, filepath):
    K = config.window
    centers = np.arange(1, len(rows) // K + 1) * K - K / 2

    rl = [row["agent1_mean_reward_per_step"] for row in rows]
    rule = [row["agent2_mean_reward_per_step"] for row in rows]
    eps = [row["agent1_epsilon"] for row in rows]
    discount = [row["agent1_discount_rate"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    axes[0].plot(range(1, len(rl) + 1), rl, alpha=0.15,
                 color=COLORS[0], label="RL, per episode")
    axes[0].plot(centers, window_means(rl, K), color=COLORS[0],
                 linewidth=2, label=f"RL, window K={K}")
    axes[0].plot(centers, window_means(rule, K), color=COLORS[1],
                 linewidth=2, label=f"Rule-based, window K={K}")
    axes[0].set_ylabel("Mean reward per step")
    axes[0].set_title("Experiment B: RL training against a rule-based agent")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(centers, window_means(discount, K), color=COLORS[2],
                 linewidth=2, label="RL discount rate (window)")
    axes[1].plot(range(1, len(eps) + 1), eps, color=COLORS[3],
                 linestyle="--", label="epsilon")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    save_figure(filepath)


def plot_q_trajectories(q_history, agent, filepath):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True)
    episodes = np.arange(1, len(q_history) + 1)

    for k, state in enumerate(agent.states()):
        index = int(np.ravel_multi_index(state, agent.state_shape))
        axes[k].plot(episodes, q_history[:, 2 * index],
                     color=COLORS[0], label="Q(s, 0) no discount")
        axes[k].plot(episodes, q_history[:, 2 * index + 1],
                     color=COLORS[1], label="Q(s, 1) discount")
        axes[k].set_title(f"s = (d={state[0]}, c={state[1]})")
        axes[k].set_xlabel("Episode")
        axes[k].grid(True, alpha=0.3)

    axes[0].set_ylabel("Q-value")
    axes[0].legend(loc="lower right", fontsize=8)
    save_figure(filepath)


def plot_policy_value(tracked, optimum, rule_value, filepath):
    episodes = [row["episode"] for row in tracked]
    values = [row["greedy_value"] for row in tracked]

    plt.figure(figsize=(10, 4.2))
    plt.plot(episodes, values, color=COLORS[0],
             label="Exact value of current greedy RL policy")
    plt.axhline(optimum, color=COLORS[2], linestyle="--",
                label=f"Best policy for state (d, c) = {optimum:.2f}")
    plt.axhline(rule_value, color=COLORS[1], linestyle=":",
                label=f"Rule-based in the same seat = {rule_value:.2f}")
    plt.xlabel("Training episode")
    plt.ylabel("Expected profit per step")
    plt.title("Greedy policy quality during training (no sampling noise)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    save_figure(filepath)


def plot_curves(curves, config, title, filepath, ylabel="Mean reward per step",
                reference=None):
    """curves: list of (label, array[seeds, windows])."""
    K = config.window

    plt.figure(figsize=(10, 5))

    for k, (label, matrix) in enumerate(curves):
        matrix = np.asarray(matrix)
        centers = np.arange(1, matrix.shape[1] + 1) * K - K / 2
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else 0 * mean

        plt.plot(centers, mean, color=COLORS[k % len(COLORS)],
                 linewidth=2, label=label)
        plt.fill_between(centers, mean - std, mean + std,
                         color=COLORS[k % len(COLORS)], alpha=0.15)

    if reference is not None:
        plt.axhline(reference[1], color="black", linestyle="--",
                    linewidth=1, label=reference[0])

    plt.xlabel("Training episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    save_figure(filepath)


def plot_bar_comparison(labels, means, stds, filepath, title,
                        ylabel="Evaluation profit per step",
                        reference=None):
    plt.figure(figsize=(9, 5))
    positions = np.arange(len(labels))

    plt.bar(positions, means, yerr=stds, capsize=6,
            color=[COLORS[k % len(COLORS)] for k in range(len(labels))])

    for x, value in zip(positions, means):
        plt.text(x, value + 0.3, f"{value:.2f}", ha="center", va="bottom")

    if reference is not None:
        plt.axhline(reference[1], color="black", linestyle="--",
                    linewidth=1, label=reference[0])
        plt.legend(loc="lower right")

    low = min(m - s for m, s in zip(means, stds))
    plt.ylim(max(0.0, low - 4), max(m + s for m, s in zip(means, stds)) + 3)
    plt.xticks(positions, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    save_figure(filepath)


# ============================================================
# 11. EXPERIMENT A + B: BASELINE AND LEARNING CURVE
# ============================================================

def run_baseline(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT A/B: BASELINE")
    print("=" * 60)

    agent, training = train_rl_against_rule(config, seed=config.seed)
    rows = training["rows"]

    write_csv(output_dir / "baseline_episode_rewards.csv", rows)

    q_path = output_dir / "q_tables.txt"
    q_path.unlink(missing_ok=True)
    save_q_table(q_path, agent, "Baseline RL agent (seed 42)")

    # Windowed learning curve (Experiment B), K = 500 for the table.
    K = 500
    window_rows = []
    rl_windows = window_means(
        [row["agent1_mean_reward_per_step"] for row in rows], K
    )
    rule_windows = window_means(
        [row["agent2_mean_reward_per_step"] for row in rows], K
    )
    discount_windows = window_means(
        [row["agent1_discount_rate"] for row in rows], K
    )
    for k in range(len(rl_windows)):
        window_rows.append({
            "window": f"{k * K + 1}-{(k + 1) * K}",
            "rl_mean_reward_per_step": rl_windows[k],
            "rule_mean_reward_per_step": rule_windows[k],
            "rl_discount_rate": discount_windows[k],
            "epsilon_at_window_start": rows[k * K]["agent1_epsilon"]
        })
    write_csv(output_dir / "baseline_training_windows.csv", window_rows)

    # Separate evaluation: RL vs rule and rule vs rule, same seed.
    eval_seed = config.seed + 500_000
    rule = RuleBasedAgent()

    rl_summary, eval_rows = evaluate_pair(
        config, [agent, rule], seed=eval_seed
    )
    rule_summary, _ = evaluate_pair(
        config, [RuleBasedAgent(), rule], seed=eval_seed
    )

    summary = []
    for run_name, run in [("RL vs Rule", rl_summary),
                          ("Rule vs Rule", rule_summary)]:
        for i, row in enumerate(run):
            agent_type = "Rule-based" if (i == 1 or run_name == "Rule vs Rule") \
                else "RL (Q-learning)"
            summary.append({"run": run_name, "agent": agent_type, **row})

    write_csv(output_dir / "baseline_evaluation.csv", summary)
    write_csv(output_dir / "baseline_evaluation_episodes.csv", eval_rows)

    optimum_table, optimum = best_response(config, rule)
    rule_exact = exact_evaluation(config, [RuleBasedAgent(), rule])
    rl_exact = exact_evaluation(config, [agent, rule])

    plot_training_rewards(rows, config, output_dir / "episode_rewards.png")
    plot_q_trajectories(training["q_history"], agent,
                        output_dir / "q_value_trajectories.png")
    plot_policy_value(training["tracked"], optimum,
                      rule_exact["reward"][0],
                      output_dir / "policy_value_during_training.png")

    print("\nSeparate evaluation results (500 episodes, seed "
          f"{eval_seed}):")
    for row in summary:
        print(
            f"{row['run']:13s} | {row['agent']:16s} | "
            f"total = {row['total_reward']:9.1f} | "
            f"mean reward/step = {row['mean_reward_per_step']:.4f} | "
            f"discount = {row['discount_actions']:5d} | "
            f"no discount = {row['no_discount_actions']:5d}"
        )

    print("\nExpected reward/step (exact): "
          f"RL greedy = {rl_exact['reward'][0]:.4f}, "
          f"rule in seat 1 = {rule_exact['reward'][0]:.4f}, "
          f"best (d, c) policy = {optimum:.4f} {optimum_table}")

    print("\nLearned greedy policy:")
    for line in format_policy(agent):
        print(line)

    print("\nBaseline training windows (K = 500):")
    for row in window_rows:
        print(
            f"{row['window']:>10s} | RL = {row['rl_mean_reward_per_step']:.3f}"
            f" | Rule = {row['rule_mean_reward_per_step']:.3f}"
            f" | RL discount = {row['rl_discount_rate']:.3f}"
        )

    return agent, optimum


# ============================================================
# 12. MULTI-SEED HELPER
# ============================================================

def run_seeds(config, make_agents, label, scenario=""):
    """
    Train and evaluate one configuration for every seed in config.seeds.

    make_agents(config, seed) -> [agent_1, agent_2]
    """

    K = config.window
    records = []
    curves = []
    discount_curves = []

    for seed in config.seeds:
        agents = make_agents(config, seed)
        training = run_training(config, agents, seed)
        rows = training["rows"]

        summary, _ = evaluate_pair(config, agents, seed=seed + 800_000)
        exact = exact_evaluation(config, agents)

        rewards = [row["agent1_mean_reward_per_step"] for row in rows]
        curves.append(window_means(rewards, K))
        discount_curves.append(window_means(
            [row["agent1_discount_rate"] for row in rows], K
        ))

        half = config.episodes // 2
        last = config.episodes - 1000

        record = {
            "scenario": scenario,
            "configuration": label,
            "seed": seed,
            "train_reward_all": float(np.mean(rewards)),
            "train_reward_last1000": float(np.mean(rewards[last:])),
            "train_discount_rate_all": float(np.mean(
                [row["agent1_discount_rate"] for row in rows]
            )),
            "eval_reward": summary[0]["mean_reward_per_step"],
            "eval_profit": summary[0]["mean_profit_per_step"],
            "eval_discount_rate": summary[0]["discount_action_rate"],
            "opponent_eval_reward": summary[1]["mean_reward_per_step"],
            "opponent_eval_discount_rate": summary[1]["discount_action_rate"],
            "exact_reward": exact["reward"][0],
            "exact_profit": exact["profit"][0],
            "policy": "".join(
                "T" if a == -1 else str(a)
                for a in agents[0].greedy_signature()
            ) if agents[0].learns else "rule",
            "policy_changes_second_half": policy_changes(
                training["signatures"][0], half
            ) if agents[0].learns else 0
        }

        if agents[1].learns:
            record["opponent_policy"] = "".join(
                "T" if a == -1 else str(a)
                for a in agents[1].greedy_signature()
            )
            record["opponent_policy_changes_second_half"] = policy_changes(
                training["signatures"][1], half
            )

        records.append(record)

    return records, np.array(curves), np.array(discount_curves)


def run_rule_seeds(config, label="Rule-based", scenario="", opponent=None):
    """Rule-based agent in seat 1 with the same evaluation seeds."""
    records = []

    for seed in config.seeds:
        other = RuleBasedAgent() if opponent is None else opponent
        agents = [RuleBasedAgent(), other]
        summary, _ = evaluate_pair(config, agents, seed=seed + 800_000)
        exact = exact_evaluation(config, agents)

        records.append({
            "scenario": scenario,
            "configuration": label,
            "seed": seed,
            "eval_reward": summary[0]["mean_reward_per_step"],
            "eval_profit": summary[0]["mean_profit_per_step"],
            "eval_discount_rate": summary[0]["discount_action_rate"],
            "exact_reward": exact["reward"][0],
            "exact_profit": exact["profit"][0]
        })

    return records


def summarize(records, label, optimum=None):
    reward = mean_std([r["eval_reward"] for r in records])
    profit = mean_std([r["eval_profit"] for r in records])
    discount = mean_std([r["eval_discount_rate"] for r in records])
    exact = mean_std([r["exact_reward"] for r in records])

    row = {
        "configuration": label,
        "n_seeds": len(records),
        "eval_reward_mean": reward[0],
        "eval_reward_std": reward[1],
        "eval_profit_mean": profit[0],
        "eval_profit_std": profit[1],
        "eval_discount_rate_mean": discount[0],
        "exact_reward_mean": exact[0],
        "exact_reward_std": exact[1]
    }

    if "train_reward_all" in records[0]:
        row["train_reward_all_mean"] = mean_std(
            [r["train_reward_all"] for r in records])[0]
        row["train_reward_last1000_mean"] = mean_std(
            [r["train_reward_last1000"] for r in records])[0]
        row["train_discount_rate_mean"] = mean_std(
            [r["train_discount_rate_all"] for r in records])[0]
        row["policy_changes_second_half_mean"] = mean_std(
            [r["policy_changes_second_half"] for r in records])[0]

    if optimum is not None:
        row["seeds_reaching_optimum"] = sum(
            r["exact_reward"] >= optimum - 1e-6 for r in records
        )

    return row


def print_summary(rows):
    for row in rows:
        text = (
            f"{row['configuration']:30s} | "
            f"eval = {row['eval_reward_mean']:.3f} +- "
            f"{row['eval_reward_std']:.3f} | "
            f"discount = {row['eval_discount_rate_mean']:.3f} | "
            f"exact = {row['exact_reward_mean']:.3f}"
        )
        if "train_reward_all_mean" in row:
            text += (
                f" | train = {row['train_reward_all_mean']:.3f}"
                f" (last 1000: {row['train_reward_last1000_mean']:.3f})"
                f" | changes = {row['policy_changes_second_half_mean']:.1f}"
            )
        if "seeds_reaching_optimum" in row:
            text += f" | optimal = {row['seeds_reaching_optimum']}/{row['n_seeds']}"
        print(text)


def rl_vs_rule(agent_class=QLearningAgent, opponent_class=RuleBasedAgent,
               **overrides):
    def make_agents(config, seed):
        return [
            agent_class.from_config(config, seed=seed + 10_000, **overrides),
            opponent_class()
        ]
    return make_agents


# ============================================================
# 13. EXPERIMENT C: LEARNING RATE
# ============================================================

def run_learning_rate_experiment(config, output_dir, optimum, cache):
    print("\n" + "=" * 60)
    print("EXPERIMENT C: LEARNING RATE (10 seeds)")
    print("=" * 60)

    records = []
    summary = []
    curves = []
    q_examples = []

    for alpha in [0.01, 0.10, 0.50]:
        label = f"alpha = {alpha:.2f}"
        result = run_seeds(config, rl_vs_rule(alpha=alpha), label)

        if alpha == config.alpha:
            cache["baseline"] = result

        records += result[0]
        summary.append({"alpha": alpha, **summarize(result[0], label, optimum)})
        curves.append((label, result[1]))

        # Q-value trajectory of one run to show update size.
        agent, training = train_rl_against_rule(
            config, seed=config.seed + 1000, alpha=alpha
        )
        q_examples.append((alpha, training["q_history"]))

    write_csv(output_dir / "learning_rate_seeds.csv", records)
    write_csv(output_dir / "learning_rate_comparison.csv", summary)
    print_summary(summary)

    plot_curves(curves, config, "Experiment C: training reward by learning rate "
                "(mean +- std over 10 seeds)",
                output_dir / "learning_rate_curves.png",
                reference=("Best expected profit for state (d, c)", optimum))

    plot_bar_comparison(
        [row["configuration"] for row in summary],
        [row["eval_reward_mean"] for row in summary],
        [row["eval_reward_std"] for row in summary],
        output_dir / "learning_rate_comparison.png",
        "Experiment C: greedy evaluation by learning rate",
        reference=("Best expected profit", optimum)
    )

    # Gap Q(s, 0) - Q(s, 1) in the low-demand state for each alpha.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    index = int(np.ravel_multi_index((0, 0), (2, 2)))
    for k, (alpha, history) in enumerate(q_examples):
        episodes = np.arange(1, len(history) + 1)
        gap = history[:, 2 * index] - history[:, 2 * index + 1]
        axes[k].plot(episodes, gap, color=COLORS[k], linewidth=0.8)
        axes[k].axhline(0, color="black", linewidth=0.8)
        axes[k].set_title(f"alpha = {alpha}: "
                          f"{np.mean(gap[2500:] < 0):.0%} of 2nd half < 0")
        axes[k].set_xlabel("Training episode")
        axes[k].grid(True, alpha=0.3)
    axes[0].set_ylabel("Q((0,0),0) - Q((0,0),1)")
    fig.suptitle("Experiment C: value gap in state (d=0, c=0); "
                 "> 0 means the greedy action is no discount")
    save_figure(output_dir / "learning_rate_q_trajectory.png")

    return summary


# ============================================================
# 14. EXPERIMENT D: EXPLORATION
# ============================================================

EXPLORATION_SETTINGS = [
    ("schedule 0.20->0.02", 0.20, 0.02, 0.995),
    ("fixed eps = 0.00", 0.00, 0.00, 1.0),
    ("fixed eps = 0.05", 0.05, 0.05, 1.0),
    ("fixed eps = 0.30", 0.30, 0.30, 1.0),
    ("schedule 0.50->0.05", 0.50, 0.05, 0.995),
]


def run_exploration_experiment(config, output_dir, optimum, cache):
    print("\n" + "=" * 60)
    print("EXPERIMENT D: EXPLORATION (10 seeds)")
    print("=" * 60)

    records = []
    summary = []
    curves = []

    for label, eps_start, eps_min, eps_decay in EXPLORATION_SETTINGS:
        if (eps_start, eps_min, eps_decay) == (
                config.epsilon_start, config.epsilon_min, config.epsilon_decay):
            result = cache["baseline"]
            result = ([{**r, "configuration": label} for r in result[0]],
                      result[1], result[2])
        else:
            result = run_seeds(
                config,
                rl_vs_rule(epsilon_start=eps_start, epsilon_min=eps_min,
                           epsilon_decay=eps_decay),
                label
            )

        records += result[0]
        summary.append({
            "exploration_setting": label,
            "epsilon_start": eps_start,
            "epsilon_min": eps_min,
            "epsilon_decay": eps_decay,
            **summarize(result[0], label, optimum)
        })
        curves.append((label, result[1]))

    write_csv(output_dir / "exploration_seeds.csv", records)
    write_csv(output_dir / "exploration_comparison.csv", summary)
    print_summary(summary)

    plot_curves(curves, config, "Experiment D: training reward by exploration "
                "setting (mean +- std over 10 seeds)",
                output_dir / "exploration_curves.png",
                reference=("Best expected profit for state (d, c)", optimum))

    plot_bar_comparison(
        [row["configuration"].replace(" ", "\n", 1) for row in summary],
        [row["eval_reward_mean"] for row in summary],
        [row["eval_reward_std"] for row in summary],
        output_dir / "exploration_comparison.png",
        "Experiment D: greedy evaluation by exploration setting",
        reference=("Best expected profit", optimum)
    )

    return summary


# ============================================================
# 15. EXPERIMENT E: FAIR MULTI-SEED COMPARISON
# ============================================================

def run_fair_comparison(config, output_dir, optimum, lr_summary,
                        exploration_summary, cache):
    print("\n" + "=" * 60)
    print("EXPERIMENT E: FAIR COMPARISON RL vs RULE (10 seeds)")
    print("=" * 60)

    rl_records = cache["baseline"][0]
    rule_records = run_rule_seeds(config)

    paired = []
    for rl, rule in zip(rl_records, rule_records):
        paired.append({
            "seed": rl["seed"],
            "rl_eval_reward": rl["eval_reward"],
            "rule_eval_reward": rule["eval_reward"],
            "difference_rl_minus_rule": rl["eval_reward"] - rule["eval_reward"],
            "rl_exact_reward": rl["exact_reward"],
            "rule_exact_reward": rule["exact_reward"],
            "rl_policy": rl["policy"],
            "rl_eval_discount_rate": rl["eval_discount_rate"],
            "rule_eval_discount_rate": rule["eval_discount_rate"]
        })
    write_csv(output_dir / "multi_seed_summary.csv", paired)

    diff_mean, diff_std = mean_std(
        [row["difference_rl_minus_rule"] for row in paired]
    )
    half_width = 2.262 * diff_std / math.sqrt(len(paired))

    def pick(rows, label):
        return next(row for row in rows if row["configuration"] == label)

    table = [
        {"agent": "RL, baseline (alpha=0.10, schedule)",
         **pick(lr_summary, "alpha = 0.10")},
        {"agent": "RL, low learning rate (alpha=0.01)",
         **pick(lr_summary, "alpha = 0.01")},
        {"agent": "RL, high learning rate (alpha=0.50)",
         **pick(lr_summary, "alpha = 0.50")},
        {"agent": "RL, fixed eps = 0.05",
         **pick(exploration_summary, "fixed eps = 0.05")},
        {"agent": "RL, fixed eps = 0.30",
         **pick(exploration_summary, "fixed eps = 0.30")},
        {"agent": "Rule-based", **summarize(rule_records, "Rule-based")},
    ]
    write_csv(output_dir / "results_table.csv", table)

    print("\nSuggested results table (evaluation, 10 seeds):")
    for row in table:
        print(f"{row['agent']:38s} | mean = {row['eval_reward_mean']:.3f} | "
              f"std = {row['eval_reward_std']:.3f} | "
              f"discount rate = {row['eval_discount_rate_mean']:.3f}")

    print(f"\nPaired difference RL - Rule: mean = {diff_mean:.4f}, "
          f"std = {diff_std:.4f}, 95% CI = [{diff_mean - half_width:.4f}, "
          f"{diff_mean + half_width:.4f}]")

    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(paired))
    plt.plot(x, [r["rl_eval_reward"] for r in paired], "o-",
             color=COLORS[0], label="RL (greedy, frozen Q-table)")
    plt.plot(x, [r["rule_eval_reward"] for r in paired], "s--",
             color=COLORS[1], label="Rule-based")
    plt.axhline(optimum, color="black", linestyle=":", linewidth=1,
                label=f"Best expected profit = {optimum:.2f}")
    plt.xticks(x, [str(r["seed"]) for r in paired])
    plt.xlabel("Training seed (evaluation seed = seed + 800,000)")
    plt.ylabel("Evaluation profit per step")
    plt.title("Experiment E: RL vs rule-based on identical demand sequences")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    save_figure(output_dir / "multi_seed_comparison.png")

    return table, (diff_mean, diff_std, half_width)


# ============================================================
# 16. QUESTION 8: TWO SIMULTANEOUSLY LEARNING AGENTS
# ============================================================

def run_simultaneous_learning(config, output_dir, cache):
    print("\n" + "=" * 60)
    print("QUESTION 8: TWO INDEPENDENT Q-LEARNERS (10 seeds)")
    print("=" * 60)

    def make_agents(config, seed):
        return [
            QLearningAgent.from_config(config, seed=seed + 10_000),
            QLearningAgent.from_config(config, seed=seed + 20_000)
        ]

    records = []
    curves_reward = []
    curves_discount = []

    for seed in config.seeds:
        agents = make_agents(config, seed)
        training = run_training(config, agents, seed)
        rows = training["rows"]

        summary, _ = evaluate_pair(config, agents, seed=seed + 800_000)
        exact = exact_evaluation(config, agents)
        half = config.episodes // 2

        records.append({
            "seed": seed,
            "agent1_policy": "".join(map(str, agents[0].greedy_signature())),
            "agent2_policy": "".join(map(str, agents[1].greedy_signature())),
            "agent1_eval_reward": summary[0]["mean_reward_per_step"],
            "agent2_eval_reward": summary[1]["mean_reward_per_step"],
            "agent1_eval_discount_rate": summary[0]["discount_action_rate"],
            "agent2_eval_discount_rate": summary[1]["discount_action_rate"],
            "agent1_exact_reward": exact["reward"][0],
            "agent2_exact_reward": exact["reward"][1],
            "agent1_policy_changes_second_half": policy_changes(
                training["signatures"][0], half),
            "agent2_policy_changes_second_half": policy_changes(
                training["signatures"][1], half)
        })

        both = [
            (row["agent1_mean_reward_per_step"]
             + row["agent2_mean_reward_per_step"]) / 2
            for row in rows
        ]
        both_discount = [
            (row["agent1_discount_rate"] + row["agent2_discount_rate"]) / 2
            for row in rows
        ]
        curves_reward.append(window_means(both, config.window))
        curves_discount.append(window_means(both_discount, config.window))

    write_csv(output_dir / "simultaneous_learning.csv", records)

    for row in records:
        print(f"seed {row['seed']:3d} | policies {row['agent1_policy']} / "
              f"{row['agent2_policy']} | eval = "
              f"{row['agent1_eval_reward']:.2f} / {row['agent2_eval_reward']:.2f}"
              f" | changes = {row['agent1_policy_changes_second_half']} / "
              f"{row['agent2_policy_changes_second_half']}")

    eval_all = [row[f"agent{i}_eval_reward"] for row in records for i in (1, 2)]
    changes = [row[f"agent{i}_policy_changes_second_half"]
               for row in records for i in (1, 2)]
    discount = [row[f"agent{i}_eval_discount_rate"]
                for row in records for i in (1, 2)]
    fixed_changes = [r["policy_changes_second_half"] for r in cache["baseline"][0]]

    summary = {
        "eval_reward_mean": mean_std(eval_all)[0],
        "eval_reward_std": mean_std(eval_all)[1],
        "eval_discount_rate_mean": mean_std(discount)[0],
        "policy_changes_second_half_mean": mean_std(changes)[0],
        "fixed_opponent_policy_changes_mean": mean_std(fixed_changes)[0]
    }
    print(f"\nRL vs RL: eval = {summary['eval_reward_mean']:.3f} +- "
          f"{summary['eval_reward_std']:.3f}, discount = "
          f"{summary['eval_discount_rate_mean']:.3f}, policy changes in second "
          f"half = {summary['policy_changes_second_half_mean']:.1f} "
          f"(RL vs fixed rule: {summary['fixed_opponent_policy_changes_mean']:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    K = config.window
    for k, (label, reward_curves, discount_curves) in enumerate([
        ("RL vs fixed rule-based (RL seat)", cache["baseline"][1],
         cache["baseline"][2]),
        ("RL vs RL (mean of both learners)", np.array(curves_reward),
         np.array(curves_discount)),
    ]):
        centers = np.arange(1, reward_curves.shape[1] + 1) * K - K / 2
        axes[0].plot(centers, reward_curves.mean(axis=0), color=COLORS[k],
                     linewidth=2, label=label)
        axes[1].plot(centers, discount_curves.mean(axis=0), color=COLORS[k],
                     linewidth=2, label=label)
    axes[0].set_ylabel("Training reward per step")
    axes[1].set_ylabel("Training discount rate")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Question 8: fixed opponent vs simultaneous learning "
                 "(mean over 10 seeds)")
    save_figure(output_dir / "simultaneous_learning.png")

    return summary


# ============================================================
# 17. QUESTION 9 AND 10: MODIFIED MARKETS AND EXTENDED AGENT
# ============================================================

def scenario_list(config):
    return [
        ("discount 20% (reference)", config, RuleBasedAgent),
        ("discount 5%", replace(config, discount_rate=0.05), RuleBasedAgent),
        ("discount 10%", replace(config, discount_rate=0.10), RuleBasedAgent),
        ("discount 30%", replace(config, discount_rate=0.30), RuleBasedAgent),
        ("revenue reward, 20%", replace(config, reward_mode="revenue"),
         RuleBasedAgent),
        ("discount 5%, tit-for-tat", replace(config, discount_rate=0.05),
         TitForTatAgent),
    ]


LEARNING_AGENTS = [
    ("RL reference (d, c)", QLearningAgent),
    ("RL extended (d, c, own)", ExtendedQLearningAgent),
    ("Model-based (proposed)", ModelBasedAgent),
]


def run_modified_markets(config, output_dir, cache):
    print("\n" + "=" * 60)
    print("QUESTION 9/10: MODIFIED MARKETS AND EXTENDED-STATE AGENT")
    print("=" * 60)

    records = []
    summary = []
    exact_rows = []

    for scenario, scenario_config, opponent_class in scenario_list(config):

        never = TablePolicyAgent(defaultdict(int), RuleBasedAgent().observe)
        always = TablePolicyAgent(defaultdict(lambda: 1),
                                  RuleBasedAgent().observe)

        reference_table, reference_best = best_response(
            scenario_config, opponent_class(), QLearningAgent)
        extended_table, extended_best = best_response(
            scenario_config, opponent_class(), ExtendedQLearningAgent)

        exact_row = {
            "scenario": scenario,
            "never_discount": exact_evaluation(
                scenario_config, [never, opponent_class()])["reward"][0],
            "always_discount": exact_evaluation(
                scenario_config, [always, opponent_class()])["reward"][0],
            "rule_based": exact_evaluation(
                scenario_config, [RuleBasedAgent(), opponent_class()])["reward"][0],
            "best_reference_state": reference_best,
            "best_reference_policy": "".join(
                str(reference_table[s]) for s in QLearningAgent().states()),
            "best_extended_state": extended_best,
            "best_extended_policy": "".join(
                str(extended_table[s]) for s in ExtendedQLearningAgent().states())
        }
        exact_rows.append(exact_row)

        agents_to_run = [("Rule-based", None)] + LEARNING_AGENTS

        for label, agent_class in agents_to_run:
            if agent_class is None:
                result = run_rule_seeds(scenario_config, label, scenario,
                                        opponent=opponent_class())
            elif (scenario_config == config and agent_class is QLearningAgent
                  and opponent_class is RuleBasedAgent):
                result = [{**r, "scenario": scenario, "configuration": label}
                          for r in cache["baseline"][0]]
            else:
                result = run_seeds(
                    scenario_config,
                    rl_vs_rule(agent_class, opponent_class),
                    label, scenario
                )[0]

            records += result
            optimum = (reference_best if agent_class is QLearningAgent
                       else extended_best)
            summary.append({
                "scenario": scenario,
                **summarize(result, label,
                            optimum if agent_class is not None else None)
            })

    write_csv(output_dir / "scenario_seeds.csv", records)
    write_csv(output_dir / "scenario_comparison.csv", summary)
    write_csv(output_dir / "scenario_exact_benchmarks.csv", exact_rows)

    print("\nExact benchmarks (expected reward per step):")
    for row in exact_rows:
        print(f"{row['scenario']:27s} | never = {row['never_discount']:.3f} | "
              f"always = {row['always_discount']:.3f} | rule = "
              f"{row['rule_based']:.3f} | best (d,c) = "
              f"{row['best_reference_state']:.3f} [{row['best_reference_policy']}]"
              f" | best (d,c,own) = {row['best_extended_state']:.3f} "
              f"[{row['best_extended_policy']}]")

    print("\nMulti-seed evaluation:")
    for row in summary:
        print(f"{row['scenario']:27s} | ", end="")
        print_summary([row])

    scenarios = [row["scenario"] for row in exact_rows]
    labels = ["Rule-based"] + [label for label, _ in LEARNING_AGENTS]
    width = 0.2
    x = np.arange(len(scenarios))

    plt.figure(figsize=(13, 5.2))
    for k, label in enumerate(labels):
        rows = [next(r for r in summary
                     if r["scenario"] == s and r["configuration"] == label)
                for s in scenarios]
        plt.bar(x + (k - 1.5) * width,
                [r["eval_profit_mean"] for r in rows], width,
                yerr=[r["eval_profit_std"] for r in rows], capsize=4,
                color=COLORS[k], label=label)
    plt.xticks(x, [s.replace(", ", "\n") for s in scenarios])
    plt.ylabel("Evaluation profit per step (mean +- std, 10 seeds)")
    plt.ylim(20, 40)
    plt.title("Question 9/10: performance when the market is modified")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper center", ncol=4)
    save_figure(output_dir / "scenario_comparison.png")

    return summary, exact_rows


# ============================================================
# 18. QUESTION 3 AND 5: WORKED CALCULATIONS
# ============================================================

def print_worked_examples(config):
    print("\n" + "=" * 60)
    print("QUESTION 3 AND 5: WORKED CALCULATIONS")
    print("=" * 60)

    q, alpha, r, gamma, max_next = 4.0, 0.1, 6.0, 0.9, 5.0
    target = r + gamma * max_next
    td_error = target - q
    print(f"Q3: target = {r} + {gamma} * {max_next} = {target:.2f}; "
          f"TD error = {td_error:.2f}; "
          f"new Q = {q} + {alpha} * {td_error:.2f} = {q + alpha * td_error:.2f}")

    for action in (0, 1):
        print(f"Q5: action {action}: price = {effective_price(config, action):.2f},"
              f" profit per sale = {profit_per_sale(config, action):.2f}")

    print("Q5: expected one-step profit E[S] * margin (competitor action b):")
    for demand in (0, 1):
        for b in (0, 1):
            values = [
                customers_for(config, demand)
                * conversion_probability(demand, a, b)
                * profit_per_sale(config, a)
                for a in (0, 1)
            ]
            print(f"    d={demand}, b={b}: no discount = {values[0]:.2f}, "
                  f"discount = {values[1]:.2f}")


# ============================================================
# 19. MAIN PROGRAM
# ============================================================

def main():
    config = Config()

    output_dir = Path(__file__).resolve().parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGENT02: RL AGENT VS RULE-BASED AGENT")
    print("=" * 60)

    print(f"Episodes: {config.episodes}")
    print(f"Steps per episode: {config.steps_per_episode}")
    print(f"Evaluation episodes: {config.evaluation_episodes}")
    print(f"Baseline seed: {config.seed}")
    print(f"Multi-seed seeds: {config.seeds}")
    print(f"Output directory: {output_dir}")

    print_worked_examples(config)

    cache = {}

    _, optimum = run_baseline(config, output_dir)

    lr_summary = run_learning_rate_experiment(
        config, output_dir, optimum, cache
    )

    exploration_summary = run_exploration_experiment(
        config, output_dir, optimum, cache
    )

    run_fair_comparison(
        config, output_dir, optimum, lr_summary, exploration_summary, cache
    )

    run_simultaneous_learning(config, output_dir, cache)

    run_modified_markets(config, output_dir, cache)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
