import numpy as np
import os
import torch
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


class Trainer:

    def __init__(self, env, env_test, algo, log_dir, device, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5, infer_reward=False):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)
        self.device = device

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.infer_reward = infer_reward

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in trange(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for i in trange(self.num_eval_episodes):
            reward_true = np.zeros(self.env_test.get_max_episode_steps())
            reward_pred = np.zeros(self.env_test.get_max_episode_steps())
            state = self.env_test.reset()
            episode_return = 0.0
            done = False
            count = 0

            while not done:
                action = self.algo.exploit(state)
                if self.infer_reward:
                    print('state: ', state.shape)
                    print('action: ', action.shape)
                    reward_hat = self.algo.disc.calculate_reward(torch.from_numpy(state).unsqueeze(0).to(self.device),
                                                                 torch.from_numpy(action).unsqueeze(0).to(self.device))
                    reward_pred[count] = reward_hat
                    self.writer.add_scalar(f'return/test/step_{step}/episode_{i}/reward_pred', reward_hat, count)

                state, reward, done, _ = self.env_test.step(action)
                reward_true[count] = reward
                episode_return += reward
                self.writer.add_scalar(f'return/test/step_{step}/episode_{i}/reward_true', reward, count)
                count += 1

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
