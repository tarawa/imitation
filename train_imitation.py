import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    device = torch.device("cuda" if args.cuda else "cpu")

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-buffer{args.buffer_size}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        infer_reward=True if args.algo != 'bc' else False,
        device=device
    )

    if args.initialize_bc_steps > 0:
        bc_algo = ALGOS['bc'](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length
        )
        trainer_bc = Trainer(
            env=env,
            env_test=env_test,
            algo=bc_algo,
            log_dir=log_dir,
            num_steps=args.initialize_bc_steps,
            eval_interval=args.initialize_bc_steps * 2,
            seed=args.seed,
            device=device
        )
        trainer_bc.train()
        trainer.algo.actor = trainer_bc.algo.actor
        trainer.algo.optim_actor = trainer_bc.algo.optim_actor

    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--initialize_bc_steps', type=int, default=0)
    args = p.parse_args()
    run(args)
