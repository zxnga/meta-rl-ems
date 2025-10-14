import argparse
from datetime import datetime
from typing import Any, Dict, List
import torch as th
from stable_baselines3 import PPO, SAC
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from src.task_generator import TaskGenerator
from src.reptile_agent import ReptileAgent
from citylearn.data import DataSet
from custom_reward import RMetaEMS
from env_utils import BatteryActionWrapper, mask_fn
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

# retain your setup_dataset_and_environment definition here...


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReptileAgent with CLI-configurable parameters")
    # Meta-learning parameters
    parser.add_argument('--inner_steps', type=int, default=100000, help='Number of inner-loop steps')
    parser.add_argument('--outer_steps', type=int, default=600, help='Number of outer-loop steps')
    parser.add_argument('--meta_lr', type=float, default=0.01, help='Meta learning rate')
    parser.add_argument('--task_batch_size', type=int, default=5, help='Number of tasks per batch')
    parser.add_argument('--save_frequency', type=int, default=1, help='How often to save meta-model')
    parser.add_argument('--experience_name', type=str, default='', help='Name prefix for saved models')

    # RL algorithm choice
    parser.add_argument('--rl_algorithm', choices=['PPO', 'MaskablePPO'], default='MaskablePPO',
                        help='RL algorithm to use')

    # rl_algo_kwargs
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', type=int, default=1)

    # inner_loop_params
    parser.add_argument('--callback', type=str, default=None, help='Callback class name or None')
    parser.add_argument('--log_interval', type=int, default=1)

    # Task generator options (hard-coded defaults kept here)
    parser.add_argument('--day_count', type=int, default=7)
    parser.add_argument('--building_count', type=int, default=1)
    parser.add_argument('--reward_class', type=str, choices=['RMetaEMS'], default='RMetaEMS')
    parser.add_argument('--central_agent', action='store_true')
    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--normalize_obs', action='store_true')
    parser.add_argument('--method', type=str, default='window')
    parser.add_argument('--exclude', type=str, default='1,17',
                        help='Comma-separated building indices to exclude')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Generate a timestamp to create a unique model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{args.experience_name}_{timestamp}"

    # Prepare task parameters
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')
    active_obs = ['month','day_type','hour','outdoor_dry_bulb_temperature',
                  'carbon_intensity','non_shiftable_load','solar_generation',
                  'electrical_storage_soc','net_electricity_consumption','electricity_pricing']
    exclude_list = [int(i) for i in args.exclude.split(',')] if args.exclude else None

    task_params = {
        'schema': schema,
        'day_count': args.day_count,
        'building_count': args.building_count,
        'reward_class': globals()[args.reward_class],
        'central_agent': args.central_agent,
        'discrete': args.discrete,
        'normalize_observation_space': args.normalize_obs,
        'active_observations': active_obs,
        'method': args.method,
        'buildings_to_exclude': exclude_list
    }

    # Build rl_algo_kwargs dynamically
    policy_map = {
        'PPO': 'MlpPolicy',  # default policy
        'MaskablePPO': MaskableActorCriticPolicy
    }
    rl_kwargs: Dict[str, Any] = {
        'n_steps': args.n_steps,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'policy': policy_map[args.rl_algorithm],
        'device': args.device,
        'verbose': args.verbose
    }

    # Build inner loop params
    inner_params: Dict[str, Any] = {
        'callback': None if args.callback in ('None', None) else globals().get(args.callback),
        'log_interval': args.log_interval
    }

    # Map algorithm string to class
    algo_map = {'PPO': PPO, 'MaskablePPO': MaskablePPO}
    chosen_algo = algo_map[args.rl_algorithm]

    # Initialize and train
    agent = ReptileAgent(
        tasks_generator_cls=TaskGenerator,
        tasks_generator_params={
            'task_callable': setup_dataset_and_environment,
            'task_callable_params': task_params,
            'revisit_ratio': 0.15,
            'revisit_start': 10,
            'sampling_method': 'random',
            'sampling_weights': None
        },
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        meta_lr=args.meta_lr,
        rl_algorithm=chosen_algo,
        re_use_actors=True,
        actor_layers=['action_net'],
        use_actor_meta_weights=False,
        task_batch_size=args.task_batch_size,
        rl_algo_kwargs=rl_kwargs,
        use_meta_optimizer=False,
        meta_optimizer=th.optim.Adam,
        inner_loop_params=inner_params,
        save_frequency=args.save_frequency,
        experience_name=unique_name
    )

    print(f"Training Meta-model for {args.outer_steps} of {args.inner_steps} each.")
    trained_model = agent.train()
    # Save with unique timestamped name
    trained_model.save(f"./meta_model_{unique_name}")
