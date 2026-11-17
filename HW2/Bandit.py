############################### LOGGER
from abc import ABC, abstractmethod
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

############################### EXPERIMENT CONSTANTS
Bandit_Reward = [1, 2, 3, 4]  # true mean reward of each advertisement
NumberOfTrials = 20000  # number of times an ad is shown

PRECISION = 1.0  # known precision tau = 1/sigma^2 of the reward noise
EPSILON_0 = 50.0  # epsilon(t) = min(1, EPSILON_0 / t)  -> decays by 1/t
M0 = 0.0  # Thompson Sampling prior mean of mu
LAMBDA_0 = 1.0  # Thompson Sampling prior precision of mu

SEED = 42  # for reproducible results
OUTPUT_DIR = "output"  # folder for CSV files and plots
SHOW_PLOTS = True  # False -> only save the plots, do not open windows


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


def _write_report(algo, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Bandit = index (0..3) of the advertisement that was shown in that trial
    df = pd.DataFrame({
        "Bandit": algo.arms,
        "Reward": algo.rewards,
        "Algorithm": algo.name,
    })
    df.to_csv(csv_path, index=False)

    rewards = np.asarray(algo.rewards)
    regrets = np.asarray(algo.regrets)
    n = len(rewards)

    logger.info(f"[{algo.name}] rewards stored in '{csv_path}'")
    logger.info(f"[{algo.name}] estimated mean of each bandit: "
                f"{np.round(algo.estimate_history[-1], 3).tolist()} "
                f"(true means: {algo.p})")
    logger.info(f"[{algo.name}] times each bandit was chosen: {algo.counts.tolist()}")
    logger.info(f"[{algo.name}] average reward over {n} trials: {rewards.mean():.4f}")
    logger.info(f"[{algo.name}] average regret over {n} trials: {regrets.mean():.4f}")
    logger.info(f"[{algo.name}] cumulative reward: {rewards.sum():.2f}")
    logger.info(f"[{algo.name}] cumulative regret: {regrets.sum():.2f}")
    return df


class Visualization():

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _finish(self, fig, filename):
        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        logger.info(f"figure saved to '{path}'")
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)

    def plot1(self, algorithms):
        for algo in algorithms:
            history = np.asarray(algo.estimate_history)  # shape (trials, bandits)
            t = np.arange(1, history.shape[0] + 1)

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, scale in zip(axes, ["linear", "log"]):
                for j in range(history.shape[1]):
                    line, = ax.plot(t, history[:, j],
                                    label=f"Bandit {j} (true mean {algo.p[j]})")
                    ax.axhline(algo.p[j], color=line.get_color(), linestyle="--", alpha=0.5)
                ax.set_xscale(scale)
                ax.set_title(f"{algo.name}: estimated mean reward ({scale} scale)")
                ax.set_xlabel("Number of trials")
                ax.set_ylabel("Estimated mean reward")
                ax.legend()
            self._finish(fig, f"plot1_{algo.name}.png")

    def plot2(self, eg, ts):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for algo in (eg, ts):
            t = np.arange(1, len(algo.rewards) + 1)
            axes[0].plot(t, np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(t, np.cumsum(algo.regrets), label=algo.name)

        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Number of trials")
            ax.legend()
        axes[0].set_ylabel("Cumulative reward")
        axes[1].set_ylabel("Cumulative regret")
        self._finish(fig, "plot2_comparison.png")


class EpsilonGreedy(Bandit):
    def __init__(self, p, tau=PRECISION, epsilon_0=EPSILON_0):
        self.p = list(p)
        self.k = len(self.p)
        self.tau = tau
        self.epsilon_0 = epsilon_0
        self.name = "EpsilonGreedy"

        self.estimates = np.zeros(self.k)  # estimated mean of each bandit
        self.counts = np.zeros(self.k, dtype=int)  # how many times each bandit was pulled

        self.arms = []  # chosen bandit in each trial
        self.rewards = []  # reward obtained in each trial
        self.regrets = []  # regret of each trial
        self.estimate_history = []  # copy of ``estimates`` after each trial
        self.num_explored = 0

    def __repr__(self):
        return f"EpsilonGreedy(true means={self.p}, epsilon(t)=min(1, {self.epsilon_0}/t))"

    def pull(self, j):
        return np.random.randn() / np.sqrt(self.tau) + self.p[j]

    def update(self, j, x):
        self.counts[j] += 1
        self.estimates[j] += (x - self.estimates[j]) / self.counts[j]

    def experiment(self):
        best = max(self.p)
        for t in range(1, NumberOfTrials + 1):
            epsilon = min(1.0, self.epsilon_0 / t)  # decay by 1/t
            if np.random.random() < epsilon:
                j = np.random.randint(self.k)  # explore
                self.num_explored += 1
            else:
                j = int(np.argmax(self.estimates))  # exploit

            x = self.pull(j)
            self.update(j, x)

            self.arms.append(j)
            self.rewards.append(x)
            self.regrets.append(best - self.p[j])
            self.estimate_history.append(self.estimates.copy())

        logger.info(f"[{self.name}] finished: explored {self.num_explored} times, "
                    f"exploited {NumberOfTrials - self.num_explored} times")

    def report(self, csv_path=os.path.join(OUTPUT_DIR, "EpsilonGreedy_rewards.csv")):
        return _write_report(self, csv_path)


class ThompsonSampling(Bandit):
    def __init__(self, p, tau=PRECISION, m0=M0, lambda0=LAMBDA_0):
        self.p = list(p)
        self.k = len(self.p)
        self.tau = tau
        self.m0 = m0
        self.lambda0 = lambda0
        self.name = "ThompsonSampling"

        self.m = np.full(self.k, float(m0))  # posterior means
        self.lambda_ = np.full(self.k, float(lambda0))  # posterior precisions
        self.sum_x = np.zeros(self.k)  # sum of rewards per bandit
        self.counts = np.zeros(self.k, dtype=int)  # pulls per bandit

        self.arms = []
        self.rewards = []
        self.regrets = []
        self.estimate_history = []  # copy of posterior means ``m`` after each trial

    def __repr__(self):
        return (f"ThompsonSampling(true means={self.p}, known precision={self.tau}, "
                f"prior N({self.m0}, 1/{self.lambda0}))")

    def pull(self, j):
        return np.random.randn() / np.sqrt(self.tau) + self.p[j]

    def sample(self):
        return np.random.randn(self.k) / np.sqrt(self.lambda_) + self.m

    def update(self, j, x):
        self.counts[j] += 1
        self.sum_x[j] += x
        self.lambda_[j] = self.lambda0 + self.tau * self.counts[j]
        self.m[j] = (self.tau * self.sum_x[j] + self.lambda0 * self.m0) / self.lambda_[j]

    def experiment(self):
        best = max(self.p)
        for _ in range(NumberOfTrials):
            j = int(np.argmax(self.sample()))  # bandit with the largest sampled mean

            x = self.pull(j)
            self.update(j, x)

            self.arms.append(j)
            self.rewards.append(x)
            self.regrets.append(best - self.p[j])
            self.estimate_history.append(self.m.copy())

        logger.info(f"[{self.name}] finished {NumberOfTrials} trials")

    def report(self, csv_path=os.path.join(OUTPUT_DIR, "ThompsonSampling_rewards.csv")):
        return _write_report(self, csv_path)


def comparison():
    np.random.seed(SEED)

    eg = EpsilonGreedy(Bandit_Reward)
    logger.info(repr(eg))
    eg.experiment()
    df_eg = eg.report()

    ts = ThompsonSampling(Bandit_Reward)
    logger.info(repr(ts))
    ts.experiment()
    df_ts = ts.report()

    # one CSV with both algorithms: {Bandit, Reward, Algorithm}
    combined_path = os.path.join(OUTPUT_DIR, "rewards.csv")
    pd.concat([df_eg, df_ts], ignore_index=True).to_csv(combined_path, index=False)
    logger.info(f"rewards of both algorithms stored in '{combined_path}'")

    viz = Visualization()
    viz.plot1([eg, ts])
    viz.plot2(eg, ts)

    # written comparison
    reward_eg, reward_ts = float(np.sum(eg.rewards)), float(np.sum(ts.rewards))
    regret_eg, regret_ts = float(np.sum(eg.regrets)), float(np.sum(ts.regrets))
    logger.info(f"Cumulative reward: EpsilonGreedy={reward_eg:.2f} | ThompsonSampling={reward_ts:.2f}")
    logger.info(f"Cumulative regret: EpsilonGreedy={regret_eg:.2f} | ThompsonSampling={regret_ts:.2f}")
    winner = "ThompsonSampling" if regret_ts < regret_eg else "EpsilonGreedy"
    logger.info(f"{winner} had the lower cumulative regret in this run "
                f"(difference: {abs(regret_eg - regret_ts):.2f})")


if __name__ == '__main__':
    logger.info("Starting the A/B testing experiment")
    comparison()
    logger.info("Done")
