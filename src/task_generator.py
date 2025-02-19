from typing import Any, Dict, Optional, Type, List, Callable, Tuple
import random
import gymnasium

ListTask = List[Tuple[gymnasium.Env,Dict[str, Any]]]

# TODO: decaying revisits

class TaskGenerator:
    def __init__(
        self,
        tasks: Optional[ListTask] = None,
        task_callable: Optional[Callable] = None,
        task_callable_params: Optional[Dict[str, Any]] = {},
        revisit_ratio = 0.15,
        revisit_start = 0,
        sampling_method = "random",
        sampling_weights: Optional[List[float]] = None):
        """
        Initialize the TaskGenerator.

        :param tasks: List of predefined tasks (optional).
        :param task_callable: Callable that generates a task given a seed (optional).
        :param revisit_ratio: Proportion of tasks to revisit in each meta-iteration (between 0 and 1).
        :param revisit_start: Number of meta_steps before being to able to revisit tasks.
        :param sampling_method: Method for sampling tasks - "cyclic", "random", or "weighted".
        :param sampling_weights: Weights of tasks if using weighted sampling method.
        """

        assert tasks is not None or task_callable is not None, \
            "Either 'tasks' (list of tasks) or 'task_callable' (callable to generate tasks) must be provided."

        self.tasks = tasks  # List of predefined tasks, if any
        self.task_callable = task_callable  # Callable to generate tasks on the fly, returns tuple(env(gymenv),info(dict))
        self.task_callable_params = task_callable_params
        self.revisit_ratio = revisit_ratio  # Proportion of tasks to revisit
        self.revisit_start = revisit_start
        self.sampling_method = sampling_method  # Task sampling method
        self.sampling_weights = sampling_weights
        self.revisit_counter = 0

        self.selected_tasks = []  # Stores generated tasks for revisiting

    def reset_history(self):
        self.selected_tasks = []
        self.revisit_counter = 0

    def get_task(self, meta_step, seed=None):
        """
        Generate or retrieve a task based on the revisit ratio and sampling method.

        :param meta_step: The current meta-iteration step, required if tasks is a list of tasks.
        :return: A task instance.

        TODO: revisit tasks when using lists of tasks, update storing of meta step using list of tasks
        """
        # Option 1: Use a predefined list of tasks
        if self.tasks is not None:
            if self.sampling_method == "cyclic":
                # Cyclically iterate through the list of tasks
                task, info = self.tasks[meta_step % len(self.tasks)]
            elif self.sampling_method == "random":
                # Randomly select a task from the list
                task, info = random.choice(self.tasks)
            elif self.sampling_method == "weighted":            
                task, info = random.choices(self.tasks, weights=self.sampling_weights, k=1)[0]
            self.selected_tasks.append({'task':task, 'task_info':info, 'meta_step': [meta_step], 'seed': None})
            return task, info, None

        # Option 2: Dynamically generate tasks using the callable
        elif self.task_callable:
            if random.random() < self.revisit_ratio and self.selected_tasks and meta_step >= self.revisit_start:
                task_idx = self._select_task_index_for_revisit()
                task = self.selected_tasks[task_idx]['task']
                info = self.selected_tasks[task_idx]['task_info']
                self.selected_tasks[task_idx]['meta_step'].append(meta_step)
                
                print(f"Revisiting task of meta_step: {self.selected_tasks[task_idx]['meta_step'][0]}")
                return task, info, self.selected_tasks[task_idx]['meta_step'][0]

            else:
                # Generate a new seed and create a new task? do not revisit inentionally
                if not seed:
                    new_seed = random.randint(0, 2**32 - 1)
                    while new_seed in [i['seed'] for i in self.selected_tasks]:
                        new_seed = random.randint(0, 2**32 - 1)
                    random.seed(new_seed)
                task, info = self.task_callable(random_seed=seed, **self.task_callable_params)
                
                # Store the seed and task for future revisits
                self.selected_tasks.append({'task':task, 'seed':new_seed, 'task_info':info, 'meta_step': [meta_step]})
                # print(f"Generated new task with seed: {new_seed}")
                return task, info, meta_step
        else:
            raise ValueError("Either 'tasks' or 'task_callable' must be provided.")
    
    def _select_task_index_for_revisit(self):
        """
        """
        if self.sampling_method == "cyclic":
            task_index = self.revisit_counter % len(self.selected_tasks)
            self.revisit_counter += 1
            return task_index
        
        elif self.sampling_method == "random":
            # Randomly select a seed from the previously generated seeds
            self.revisit_counter += 1
            return random.randint(0, len(self.selected_tasks)-1)
        
        elif self.sampling_method == "weighted":
            # return random.choices(self.tasks, weights=self.sampling_weights, k=1)[0]
            self.revisit_counter += 1
            raise NotImplementedError