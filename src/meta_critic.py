from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from src.task_generator import TaskGenerator


class TaskActionEncoderNetwork(nn.Module):
    # input_dim k tuples
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(TaskActionEncoderNetwork, self).__init__()
        print(input_dim, hidden_dim, )

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden_state = hidden[-1]  # Shape: (batch_size, hidden_dim)
        output = self.fc(last_hidden_state)  # Shape: (batch_size, output_dim)
        return output

class MetaValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch=[80, 80], activation_fn=nn.ReLU):
        """
        Parameters:
        - input_dim: int, number of input features
        - output_dim: int, number of output neurons
        - net_arch: list of int, number of neurons in each hidden layer
        - activation_fn: callable, activation function class (e.g., nn.ReLU, nn.Sigmoid)
        """
        super(MetaValueNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, net_arch=[64,64], activation_fn=nn.ReLU):
        super(Actor, self).__init__()
        layers = []
        prev_dim = input_dim
        for i,hidden_dim in enumerate(net_arch):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class MetaCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        reward_dim,
        taen_hidden_dim=30,
        taen_output_dim=3,
        taen_num_layers=1,
        mvn_hidden_dims=[80, 80],
        mvn_activation_fn=nn.ReLU
        ):
        """
        MetaCritic combines the Task-Action Encoder Network (TAEN) and Meta Value Network (MVN).
        Args:
            state_dim (int): Dimension of the state input.
            action_dim (int): Dimension of the action input.
            reward_dim (int): Dimension of the reward input.
            taen_hidden_dim (int): Hidden size for the LSTM in TAEN.
            taen_output_dim (int): Output dimension of the TAEN (task-actor embedding size).
            mvn_hidden_dims (list): Hidden layer dimensions for the MVN.
            mvn_activation_fn (callable): Activation function for MVN layers.
        """
        super(MetaCritic, self).__init__()
        
        # Input dimension for TAEN
        taen_input_dim = state_dim + action_dim + reward_dim

        self.taen = TaskActionEncoderNetwork(input_dim=taen_input_dim,
                                             output_dim=taen_output_dim, 
                                             hidden_dim=taen_hidden_dim,
                                             num_layers=taen_num_layers)
        
        # Input dimension for MVN (state + action + task-actor embedding)
        mvn_input_dim = state_dim + action_dim + taen_output_dim

        # Meta Value Network
        self.mvn = MetaValueNetwork(input_dim=mvn_input_dim, 
                                    output_dim=1,  # Q-value output
                                    net_arch=mvn_hidden_dims, 
                                    activation_fn=mvn_activation_fn)

    def forward(self, states, actions, taen_input):
        """
        Forward pass through MetaCritic.
        Args:
            states (torch.Tensor): Batch of states (batch_size, state_dim).
            actions (torch.Tensor): Batch of actions (batch_size, action_dim).
            rewards (torch.Tensor): Batch of rewards (batch_size, k, reward_dim).
        Returns:
            q_values (torch.Tensor): Predicted Q-values (batch_size, 1).
        """
        # taen_input = th.cat([states, actions, rewards], dim=-1)  # Shape: (batch_size, k, input_dim)
        task_actor_embedding = self.taen(taen_input)  # Shape: (batch_size, taen_output_dim)
        mvn_input = th.cat([states, actions, task_actor_embedding], dim=-1)  # Shape: (batch_size, mvn_input_dim)
        q_values = self.mvn(mvn_input)  # Shape: (batch_size, 1)
        
        return q_values, task_actor_embedding

class MetaRLAlgorithm:
    def __init__(
        self,
        tasks_generator_cls: TaskGenerator,
        tasks_generator_params: Dict[str, Any],
        task_embedding_arch_params: Dict[str,int],
        task_embedding_window: int,
        actor_arch=[64,64], 
        meta_critic_arch=[80, 80],
        meta_critic_act_fn = nn.ReLU,
        reward_net_arch = [64,64],
        reward_net_act_fn = nn.ReLU,
        buffer_size=150_000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        intermediate_saves: int=0
        ):
        """
            Initializes the MetaRLAlgorithm class.
            Args:
                state_dim (int): Dimension of the state space.
                action_dim (int): Dimension of the action space.
                reward_dim (int): Dimension of the reward (usually 1).
                embedding_dim (int): Dimension of the task-actor embedding.
                hidden_dim (int): Number of hidden units for actor and TAEN.
                meta_critic_hidden_dims (list): Hidden layers for the Meta-Critic.
                buffer_size (int): Maximum size of each task's replay buffer.
                batch_size (int): Number of samples per training batch.
                gamma (float): Discount factor for TD learning.
                lr (float): Learning rate for optimizers.
                num_tasks (int): Number of tasks in the meta-training phase.
        """
        self.task_embedding_arch_params = task_embedding_arch_params
        self.task_embedding_window = task_embedding_window
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.meta_critic_arch = meta_critic_arch
        self.meta_critic_act_fn = meta_critic_act_fn
        self.actor_arch = actor_arch
        self.reward_net_arch = reward_net_arch
        self.reward_net_act_fn = reward_net_act_fn

        self.task_generator_cls = tasks_generator_cls
        self.tasks_generator_params = tasks_generator_params
        self.task_generator = self.instanciate_task_generator()
        self.observation_space, self.action_space, self.reward_dim = self._get_task_dims()
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        
        # Initialize Meta-Critic
        self.meta_critic = MetaCritic(
            state_dim=self.state_dim, action_dim=self.action_dim, reward_dim=self.reward_dim,
            mvn_hidden_dims=meta_critic_arch,mvn_activation_fn=meta_critic_act_fn,
            **task_embedding_arch_params)
        self.meta_critic_optimizer = torch.optim.Adam(self.meta_critic.parameters(), lr=lr)
        
        # Initialize Actor
        self.actor = Actor(self.state_dim, self.action_dim, self.actor_arch)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # self.reward_network = MetaValueNetwork(input_dim=self.state_dim+self.action_dim, 
        #                             output_dim=1,  # reward output
        #                             net_arch=reward_net_arch, 
        #                             activation_fn=reward_net_act_fn)

    def _get_task_dims(self):
        task, _ = self.task_generator.get_task(0)
        observation_space = task.observation_space
        action_space = task.action_space
        _ = task.reset()
        _, r, _, _, _ = task.step(action_space.sample())
        self.task_generator.reset_history()
        if isinstance(r, list) or isinstance(r, np.ndarray):
            reward_dim = len(r)
        else:
            reward_dim = 1
        return observation_space, action_space, reward_dim

    def instanciate_task_generator(self):
        return self.task_generator_cls(**self.tasks_generator_params)
        
    def _instanciate_buffer(self, outer_steps):
        return ReplayBuffer(
            buffer_size=self.buffer_size*outer_steps, observation_space=self.observation_space,
            action_space=self.action_space, n_envs=outer_steps)

    def collect_trajectories(self, task_id, task_env, trajectory_length=200):
        """
        Collect trajectories dynamically during training and store them in the task's replay buffer.
        Args:
            task_id (int): ID of the current task.
            task_env: Task-specific environment.
            trajectory_length (int): Number of steps to collect per trajectory.
        """
        state, _ = task_env.reset()
        for _ in range(trajectory_length):
            # Actor generates action
            action = self.actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy()
            next_state, reward, done, _, _ = task_env.step(action)
            self.replay_buffers[task_id].add((state, action, reward, next_state))
            state = next_state

            if done:
                state, _ = task_env.reset()

    def update_history(self,history, next_states, next_actions, next_rewards):
        new_h = torch.cat((next_states,next_actions, next_rewards), axis=1)
        updated_history = torch.cat((history[:, 1:], new_h.unsqueeze(1)), dim=1)
        return updated_history
    
    def update_meta_critic(self, task_id):
        """
        Update the Meta-Critic using the Temporal Difference (TD) error.
        Args:
            task_id (int): ID of the current task.
        """
        if len(self.replay_buffers.pos[task_id]) < self.batch_size:
            return
        
        # Sample batch
        states, actions, next_states, dones, rewards, history = self.replay_buffers.sample(self.batch_size, task_id)
        
        # Compute Q-values and TD target
        q_values, z_t = self.meta_critic(states, actions, history)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_rewards = torch.zeros_like(next_actions) #use placeholder for next_rewards as it is unknown for action taken using actor
            updated_history = self.update_history(history, next_states, next_actions, next_rewards)
            next_q_values, _ = self.meta_critic(next_states, next_actions, updated_history)
            td_target = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Meta-Critic loss
        meta_critic_loss = F.mse_loss(q_values, td_target)
        
        # Optimize Meta-Critic
        self.meta_critic_optimizer.zero_grad()
        meta_critic_loss.backward()
        self.meta_critic_optimizer.step()
        
        return meta_critic_loss.item()

    def update_actor(self, task_id):
        """
        Update the Actor to maximize the Q-values predicted by the Meta-Critic.
        Args:
            task_id (int): ID of the current task.
        """
        if len(self.replay_buffers.pos[task_id]) < self.batch_size:
            return
        
        # Sample batch
        states, _, _, _, _, history = self.replay_buffers.sample(self.batch_size, task_id)
         
        # Compute Actor loss
        predicted_actions = self.actor(states)
        q_values, _ = self.meta_critic(states, predicted_actions, torch.zeros_like(states[task_id, :1]))
        actor_loss = -q_values.mean()
        
        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()

    def learn(self, task_envs, num_iterations=1000, trajectory_length=10):
        """
        Main training loop for the meta-RL algorithm.
        Args:
            task_envs (list): List of task-specific environments.
            num_iterations (int): Number of training iterations.
            trajectory_length (int): Number of steps per trajectory.
        """
        for iteration in range(num_iterations):
            # Step 1: Collect trajectories for each task
            for task_id, task_env in enumerate(task_envs):
                self.collect_trajectories(task_id, task_env, trajectory_length)

            # Step 2: Update Meta-Critic and Actor
            meta_critic_losses, actor_losses = [], []
            for task_id in range(self.num_tasks):
                meta_critic_loss = self.update_meta_critic(task_id)
                actor_loss = self.update_actor(task_id)
                if meta_critic_loss is not None:
                    meta_critic_losses.append(meta_critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)

            # Logging
            avg_meta_critic_loss = sum(meta_critic_losses) / len(meta_critic_losses) if meta_critic_losses else 0
            avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
            print(f"Iteration {iteration}, Meta-Critic Loss: {avg_meta_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}")

    
    def learn(self, task_envs, num_iterations=1000, trajectory_length=10):
        """
        Main training loop for the meta-RL algorithm.
        Args:
            task_envs (list): List of task-specific environments.
            num_iterations (int): Number of training iterations.
            trajectory_length (int): Number of steps per trajectory.
        """
        for iteration in range(num_iterations):
            # Step 1: Collect trajectories for each task
            batch_task = sample_tasks()
            for task_id, task_env in enumerate(batch_task):
                self.collect_trajectories(task_id, task_env, trajectory_length)

            # Step 2: Update Meta-Critic and Actor
            meta_critic_losses, actor_losses = [], []
            for task_id in range(self.num_tasks):
                meta_critic_loss = self.update_meta_critic(task_id)
                actor_loss = self.update_actor(task_id)
                if meta_critic_loss is not None:
                    meta_critic_losses.append(meta_critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)

            # Logging
            avg_meta_critic_loss = sum(meta_critic_losses) / len(meta_critic_losses) if meta_critic_losses else 0
            avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
            print(f"Iteration {iteration}, Meta-Critic Loss: {avg_meta_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}")
