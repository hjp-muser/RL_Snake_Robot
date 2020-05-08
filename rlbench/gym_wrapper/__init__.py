from gym.envs.registration import register
import rlbench.backend.task as task
import os
from rlbench.utils.string_methods import name_to_task_class
from rlbench.gym_wrapper.gym_env import RLBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym_wrapper:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        }
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym_wrapper:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )
    register(
        id='%s-both-v0' % task_name,
        entry_point='rlbench.gym_wrapper:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'both'
        }
    )
    register(
        id='%s-state-param-v0' % task_name,
        entry_point='rlbench.gym_wrapper:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'action_mode': 'trigon',
            'observation_mode': 'state'
        }
    )
