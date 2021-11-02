import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import StateDependentPolicy


class BC(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        self.learning_steps_disc = 0
        self.optim_bc = Adam(self.actor.parameters(), lr=lr_disc)
        self.loss_bc = nn.MSELoss()
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from expert's demonstrations.
            states_exp, actions_exp = \
                self.buffer_exp.sample(self.batch_size)[:2]
            # Update BC Agent.
            self.update_bc(states_exp, actions_exp, writer)

    def update_bc(self, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        self.optim_bc.zero_grad()
        actions, logpi = self.actor.sample(states_exp)
        print('actions: ', actions_exp.shape)
        loss = self.loss_bc(actions, actions_exp)
        loss.backward()
        self.optim_bc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/bc', loss.item(), self.learning_steps)

            writer.add_scalar('stats/log_pi', logpi.mean().item(), self.learning_steps)
