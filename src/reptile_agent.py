from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union, List, Callable

import os
import math
import numpy as np
import torch as th
import copy
from tqdm.notebook import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .task_generator import TaskGenerator
from .utils import (
    get_unique_experience_name,
    load_weights_from_source,
    extract_layer_weights)


class ReptileAgent:
    def __init__(
        self,
        tasks_generator_cls: TaskGenerator,
        tasks_generator_params: Dict[str, Any],
        inner_steps: int,
        outer_steps: int,
        meta_lr: float, # meta_lr
        rl_algorithm: Union[th.nn.Module],
        re_use_actors: bool = False, # store actors, and if same task is sampled, use previously trained actor
        actor_layers: List[str] = [],
        use_actor_meta_weights: bool = True, # use meta weights for actor initialization when not re-using actor or when task not seen
        task_batch_size: int = 1,
        rl_algo_kwargs: Optional[Dict[str, Any]] = {},
        ignored_layers: List[str] = [], # layers of the rl_algorithm to ignore when updating the meta model parameters
        use_meta_optimizer: bool = False,
        meta_optimizer: th.optim = th.optim.Adam,
        inner_loop_params: Optional[Dict[str, Any]] = {},
        save_frequency: int = 1,
        meta_weights_dir: str = ('./meta_policy_weights'),
        tensorboard_logs: Optional[str] = './inner_loop_logs',
        experience_name: str = '',
        ):
        assert hasattr(rl_algorithm, 'learn'), f'RL algorithm needs a .learn() method to train inner loop.'
        assert task_batch_size > 0, f"task_batch_size must be > 0, got {task_batch_size}"
        assert not re_use_actors or actor_layers, "actor_layers is required when re_use_actors is True"
        if 'n_steps' in rl_algo_kwargs:
            if 'n_steps' in rl_algo_kwargs and inner_steps > rl_algo_kwargs['n_steps']:
                print(f"Warning inner_steps({inner_steps}) > n_steps({rl_algo_kwargs['n_steps']}) ! A supplementary round of updates will be done.")
                rl_algo_kwargs['n_steps'] = inner_steps
            else:
                rl_algo_kwargs['n_steps'] = inner_steps
            rl_algo_kwargs['n_steps'] = inner_steps      

        self.task_generator_cls = tasks_generator_cls
        self.tasks_generator_params = tasks_generator_params
        self.task_generator = self.instanciate_task_generator()

        self.inner_steps = inner_steps
        self.inner_loop_params = inner_loop_params
        self.outer_steps = outer_steps
        self.meta_lr = meta_lr
        self.task_batch_size = task_batch_size
        if task_batch_size > 1:
            prev_outer_steps = outer_steps
            self.outer_steps = math.ceil(outer_steps / task_batch_size)
            print(f'Carefull ! Task_batch_size > 1. Total number of outer loop steps are: {prev_outer_steps}/{task_batch_size}={self.outer_steps} !')
            

        self.rl_algorithm = rl_algorithm
        self.rl_algo_kwargs = rl_algo_kwargs
        self.ignored_layers = ignored_layers
        self.re_use_actors = re_use_actors
        self.actor_layers = actor_layers
        self.use_actor_meta_weights = use_actor_meta_weights

        #TODO: if no task generator
        self.meta_algo = self.instanciate_model(self.task_generator.get_task(0)[0], False)
        self.task_generator.reset_history()
        self.meta_policy = self.meta_algo.policy #ActorCriticPolicy
        self.meta_optimizer = meta_optimizer(
            self.meta_policy.parameters(),
            lr=self.meta_lr)
        self.use_meta_optimizer = use_meta_optimizer
        self.save_frequency = save_frequency

        self.gradient_norms = np.zeros(outer_steps)
        self.parameter_trajectory = np.zeros((outer_steps, sum(p.numel() for p in self.meta_policy.parameters())))
        self.layer_trajectories = {
            name: np.zeros((outer_steps, param.numel()))
            for name, param in self.meta_policy.named_parameters()
        }

        self.ignored_params, unmatched_layers = self.get_model_parameters_from_name(self.ignored_layers)
        assert not unmatched_layers, (
            f"Layers to ignored during meta-learning stage not found in the RL algorithm: {unmatched_layers}"
        )

        self.meta_weights_dir = meta_weights_dir
        self.experience_name = get_unique_experience_name(experience_name, self.meta_weights_dir)
        self.tensorboard_logs = os.path.join(tensorboard_logs,self.experience_name)
        print(f"Meta-weights saved at: {self.meta_weights_dir}")
        print(f"Inner-loop logs saved at: {self.tensorboard_logs}")

        print(f"Total number of timesteps in the env is: {outer_steps*inner_steps:_}")

    def save_meta_weights(self, meta_iteration: int):
        meta_weights_dir = os.path.join(self.meta_weights_dir, self.experience_name)
        os.makedirs(meta_weights_dir, exist_ok=True)
        save_path = os.path.join(meta_weights_dir, f"meta_policy_step_{meta_iteration + 1}.pth")
        th.save(self.meta_policy.state_dict(), save_path)
    
    def get_model_parameters_from_name(self, target_layers):
        """
            Get fully qualified parameter names based on specified layer names.

            :return: List of fully qualified parameter names
                    List of unmatched layers.
        """
        all_params = {name for name, _ in self.meta_policy.named_parameters()}
        target_parameters = [name for name in all_params if any(name.startswith(layer) for layer in target_layers)]
        unmatched_layers = [layer for layer in target_layers if not any(name.startswith(layer) for name in all_params)]

        return target_parameters, unmatched_layers

    def instanciate_task_generator(self):
        return self.task_generator_cls(**self.tasks_generator_params)

    def instanciate_model(self, task, inner):
        policy = self.rl_algo_kwargs.get('policy', 'MlpPolicy')
        algo_kwargs = {k: v for k, v in self.rl_algo_kwargs.items() if k != 'policy'}
        if inner and self.tensorboard_logs is not None:
            algo_kwargs['tensorboard_log'] = self.tensorboard_logs
        return self.rl_algorithm(env=task, policy=policy, **algo_kwargs)
        
    def check_tasks_homogeneity(self):
        tasks = [t[1] for t in self.tasks]
        action_space = all(t.action_space == tasks[0].action_space for t in tasks)
        obs_space = all(t.observation_space.shape == tasks[0].observation_space.shape for t in tasks)
        return all([action_space, obs_space])

    def reptile_update(self, task_models, ignore_layers=None):
        """
            Perform the Reptile meta-update with a batch of task-specific models.

            :param task_models: List of task-specific models (one for each task in the batch).
            :param task_batch_size: Number of tasks in the batch.
            :param ignore_layers: List of parameter names (strings) to ignore during the update.
        """
        # Get original meta-parameters
        original_params = self.meta_policy.state_dict()

        # Initialize a dictionary to store the accumulated differences
        accumulated_deltas = {name: th.zeros_like(param, device=param.device) for name, param in original_params.items()}

        # Accumulate the parameter differences for each task in the batch
        for task_model in task_models:
            task_params = task_model.policy.state_dict()
            for name in original_params:
                # Accumulate the difference for this parameter
                if name in self.ignored_params:  # Skip ignored layers
                    continue
                if self.use_meta_optimizer and isinstance(self.meta_optimizer, th.optim.Optimizer):
                    accumulated_deltas[name] += (original_params[name] - task_params[name])
                else:
                    accumulated_deltas[name] += (task_params[name] - original_params[name]) / self.task_batch_size

        # If using a meta-optimizer, apply the averaged update as pseudo-gradients
        if self.use_meta_optimizer and isinstance(self.meta_optimizer, th.optim.Optimizer):
            self.meta_optimizer.zero_grad()  # Reset gradients

            # Set each parameter's gradient as the averaged difference
            for name, param in self.meta_policy.named_parameters():
                if name in accumulated_deltas and name not in self.ignored_params:  # Skip ignored layers:
                    param.grad = accumulated_deltas[name].detach()  # Set pseudo-gradient

            # Apply the meta-update with the meta-optimizer
            self.meta_optimizer.step()

        else:
            # Directly apply the update without a meta-optimizer
            updated_params = {}
            for name, param in self.meta_policy.named_parameters():
                if name in accumulated_deltas and name not in self.ignored_params:  # Skip ignored layers:
                    # Update each parameter using the averaged difference
                    updated_params[name] = (param + self.meta_lr * accumulated_deltas[name]).detach()

            # Load the updated parameters back into the meta-policy
            self.meta_policy.load_state_dict(updated_params)

    def update_to_task(self, task_model):
        task_model.learn(self.inner_steps, **self.inner_loop_params)

    def track_parameter_trajectory(self, meta_iteration):
        """
        Record the current parameters of the model at each meta-iteration.
        """
        # Flatten and store all parameters into a single vector for each meta-iteration
        params = np.concatenate([param.data.cpu().numpy().flatten() for param in self.meta_policy.parameters()])
        self.parameter_trajectory[meta_iteration] = params

    def track_gradient_norm(self, meta_iteration):
        """
        Calculate and track the norm of the gradients of the meta-policy's parameters.
        """
        total_norm = 0
        for param in self.meta_policy.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms[meta_iteration] = total_norm
    
    def track_layer_parameters_trajectory(self, meta_iteration):
        """
        Track parameters of each layer at each meta-iteration.
        """
        # Store parameters for each layer dynamically
        for name, param in self.meta_policy.named_parameters():
            self.layer_trajectories[name][meta_iteration] = param.data.cpu().numpy().flatten()

    def reduce_parameter_trajectory(self, method='pca', dimensions=2):
        """
        Reduce the parameter trajectory using PCA, t-SNE, or UMAP.
        
        :param method: Decomposition method ('pca', 'tsne', 'umap').
        :param dimensions: 2 or 3 for visualization dimensionality.
        :return: Matplotlib figure object for further manipulation or saving.
        """
        assert method in ['pca', 'tsne', 'umap'], "Invalid method; choose 'pca', 'tsne', or 'umap'."
        assert dimensions in [2, 3], "Only 2D or 3D visualizations are supported."

        if method == 'pca':
            reducer = PCA(n_components=dimensions)
        elif method == 'tsne':
            reducer = TSNE(n_components=dimensions)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=dimensions)
        else:
            raise ValueError("Unsupported method. Choose 'PCA', 't-SNE', or 'UMAP'.")

        # Perform the dimensionality reduction
        reduced_trajectory = reducer.fit_transform(self.parameter_trajectory)
        return reduced_trajectory

    def train(self, reset_task_history_before_learning=True):
        """
            Train the meta-model using Reptile.

            1. Sample tasks
            2. Setup Inner Algo
            2. Train inner policy using SB3
            3. Copy the weights to the meta policy using update rule

            Args:
                reset_task_history_before_learning (bool): Whether to reset the task history before learning.
        """
        actors = {}
        # in case we want to continue the training and keep the task history for revisit set to false
        if reset_task_history_before_learning:
            self.task_generator.reset_history()
        
        for meta_iteration in tqdm(range(self.outer_steps), desc="Meta-training progress"):
            # print(meta_iteration)
            task_models = []
            task_batch = [
                self.task_generator.get_task(meta_iteration*self.task_batch_size+i)
                for i in range(self.task_batch_size)
            ]
            
            for current_task, task_info, first_occurence in task_batch:                
                # 1. load meta weights into task specific model
                exclude_layers = []
                task_model = self.instanciate_model(current_task, True)

                if not self.use_actor_meta_weights and self.actor_layers:
                    # we exclude actor weights from initialization (use random weights)
                    exclude_layers, _ = self.get_model_parameters_from_name(self.actor_layers)
                load_weights_from_source(self.meta_policy, task_model.policy, exclude_layers, detach=True)

                if self.re_use_actors:
                    task_actor = actors.get(first_occurence)
                    if task_actor:
                    # 2. load actor weights
                        load_weights_from_source(task_actor, task_model.policy, detach=True)

                self.update_to_task(task_model)     # Perform inner loop adaptation on the task
                task_models.append(task_model)
                if self.re_use_actors:
                    # 3. store actor
                    actors[first_occurence] = extract_layer_weights(task_model.policy, self.actor_layers, detach=True)
            
            # Perform the Reptile update based on the entire batch of tasks
            self.reptile_update(task_models)
            if self.use_meta_optimizer:
                self.track_gradient_norm(meta_iteration)
            self.track_parameter_trajectory(meta_iteration)
            self.track_layer_parameters_trajectory(meta_iteration)

            if (meta_iteration + 1) % self.save_frequency == 0:
                self.save_meta_weights(meta_iteration)
        self.save_meta_weights(meta_iteration)
        return self.meta_algo