from pysc2.env import sc2_env
from multiprocessing import Pipe, Process
from pysc2.env.environment import StepType
from absl import flags
import numpy as np
import sys

def wrap(*values):
    return [[value] for value in values]


class SCEnvironmentWrapper:
    def __init__(self, interface, env_kwargs):
        flags.FLAGS(sys.argv)
        self.env = sc2_env.SC2Env(**env_kwargs)
        self.render = env_kwargs['visualize']
        self.interface = interface
        self.done = False
        self.timestep = None
        self.num_parallel_instances = 1

    def step(self, action_list):
        """
        :param action_list:
            List of pysc2 actions.
        :return:
            env_state: The state resulting after the action has been taken.
            total_reward: The accumulated reward from the environment
            done: Whether the action resulted in the environment reaching a terminal state.
        """
        if self.done:
            dummy_state, dummy_mask = self.interface.dummy_state()
            return [dummy_state], [dummy_mask], np.nan, [int(self.done)]

        total_reward = 0
        for action in action_list[0]:
            self.timestep = self.env.step([action])[0]
            # if self.render:
            #     time.sleep(0.15)

            total_reward += self.timestep.reward
            self.done = int(self.timestep.step_type == StepType.LAST)

            if self.done:
                break

        state, action_mask = self.interface.convert_state(self.timestep)
        return wrap(state, action_mask, total_reward, int(self.done))

    def reset(self):
        timestep = self.env.reset()[0]
        state, action_mask = self.interface.convert_state(timestep)
        self.timestep = timestep
        self.done = False
        return wrap(state, action_mask,  0, int(self.done))

    def close(self):
        self.env.__exit__(None, None, None)
