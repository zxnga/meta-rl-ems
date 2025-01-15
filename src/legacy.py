# TODO: enable the sazmpling of batch of tasks and averaging the weights of each
# tasks before moving the meta parameters in this direction

class MetaMaskPPO_:
    def __init__(
        self,
        tasks_generator_cls: TaskGenerator,
        tasks_generator_params: Dict[str, Any],
        inner_steps: int,
        outer_steps: int,
        meta_lr: float, # meta_lr
        rl_algorithm: Union[OnPolicyAlgorithm, OffPolicyAlgorithm, th.nn.Module],
        task_batch_size: int = 1,
        rl_algo_kwargs: Optional[Dict[str, Any]] = {},
        use_meta_optimizer: bool = False,
        meta_optimizer: th.optim = th.optim.Adam,
        inner_loop_params: Optional[Dict[str, Any]] = {},
        ):

        assert hasattr(rl_algorithm, 'learn'), f'RL algorithm needs a .learn() method to train inner loop.'
        assert task_batch_size > 0, f"task_batch_size must be > 0, got {task_batch_size}"
        if 'n_steps' in rl_algo_kwargs and rl_algo_kwargs['n_steps'] != inner_steps:
            print(f"inner_steps ({inner_steps}) and n_steps ({rl_algo_kwargs['n_steps']}) should be the same, using inner_steps !")
            rl_algo_kwargs['n_steps'] = inner_steps
        else:
             rl_algo_kwargs['n_steps'] = inner_steps

        self.task_generator_cls = tasks_generator_cls
        self.tasks_generator_params = tasks_generator_params
        self.task_generator = self.instanciate_task_generator()

        self.inner_steps = inner_steps
        self.inner_loop_params = inner_loop_params
        self.outer_steps = outer_steps
        self.meta_lr = meta_lr
        self.task_batch_size = task_batch_size

        self.rl_algorithm = rl_algorithm
        self.rl_algo_kwargs = rl_algo_kwargs

        self.meta_algo = self.instanciate_model(self.task_generator.get_task(0)[0])
        self.task_generator.reset_history()
        self.meta_policy = self.meta_algo.policy #ActorCriticPolicy
        self.meta_optimizer = meta_optimizer(
            self.meta_policy.parameters(),
            lr=self.meta_lr)
        self.use_meta_optimizer = use_meta_optimizer

    def instanciate_task_generator(self):
        return self.task_generator_cls(**self.tasks_generator_params)

    def instanciate_model(self, task):
        policy = self.rl_algo_kwargs.get('policy', 'MlpPolicy')
        algo_kwargs = {k: v for k, v in self.rl_algo_kwargs.items() if k != 'policy'}
        return self.rl_algorithm(env=task, policy=policy, **algo_kwargs)
        
    def check_tasks_homogeneity(self):
        tasks = [t[1] for t in self.tasks]
        action_space = all(t.action_space == tasks[0].action_space for t in tasks)
        obs_space = all(t.observation_space.shape == tasks[0].observation_space.shape for t in tasks)
        return all([action_space, obs_space])

    def reptile_update(self, task_model):
        original_params = self.meta_policy.state_dict()
        task_params = task_model.policy.state_dict()

        # Check if we're using a meta-optimizer for the update
        if self.use_meta_optimizer and isinstance(self.meta_optimizer, th.optim.Optimizer):
            # Use the meta-optimizer to update parameters
            self.meta_optimizer.zero_grad()  # Reset gradients

            # Calculate "pseudo-gradients" based on the difference for each parameter
            for name, param in self.meta_policy.named_parameters():
                # Set the gradient manually to the Reptile update difference
                param.grad = task_params[name] - param

            # Step with the meta-optimizer, applying the meta-learning rate
            self.meta_optimizer.step()

        else:
            # Direct Reptile update without a meta-optimizer
            updated_params = {}
            for param_name in original_params:
                # Perform the Reptile update for each parameter directly
                updated_params[param_name] = original_params[param_name] + \
                                            self.meta_lr * (task_params[param_name] - original_params[param_name])

            # Load the updated parameters into the meta-policy
            self.meta_policy.load_state_dict(updated_params)

    def update_to_task(self, task_model):
        task_model.learn(self.inner_steps, **self.inner_loop_params)

    def train(self):
        """
            1. Sample tasks
            2. Setup Inner Algo
            2. Train inner policy using SB3
            3. Copy the weights to the meta policy using update rule
        """
        for meta_iteration in tqdm(range(self.outer_steps), desc="Meta-training progress"):
            # Sample a task for this meta-iteration
            current_task, task_info = self.task_generator.get_task(meta_iteration)
            # print(task_info)
            _ = current_task.reset() #to make sure
            
            # Save the current meta-parameters (global parameters)
            original_params = copy.deepcopy(self.meta_policy.state_dict())
            
            # Inner loop: Task-specific adaptation
            task_model = self.instanciate_model(current_task)
            task_model.policy.load_state_dict(original_params)
            # task_model = copy.deepcopy(self.meta_algo) 
            
            self.update_to_task(task_model)
            
            # After task iterations, perform Reptile meta-update
            self.reptile_update(task_model)
        return self.meta_algo