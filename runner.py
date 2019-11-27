from multiprocessing import set_start_method
from absl import app
import absl as flags

from pysc2.env import sc2_env
from pysc2.lib import actions

from agent import LSTMAgent
from env_interface import EmbeddingInterfaceWrapper, TrainMarines, BeaconEnvironmentInterface
from environment import MultipleEnvironment, SCEnvironmentWrapper
from impala_master import Learner
from normal_learner import ActorCriticLearner

env_kwargs = {
    # "map_name": "EconomicRLTraining",
    # "map_name": "StalkersVsRoaches",
    "map_name": "MoveToBeacon",
    # "map_name": "CollectMineralShards",
    # "map_name": "BuildMarines",
    "visualize": False,
    "step_mul": 8,
    'game_steps_per_episode': None,
    'save_replay_episodes': 0,
    'replay_dir': 'saves/',
    "agent_interface_format": sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=84,
            minimap=84),
        action_space=actions.ActionSpace.FEATURES,
        use_feature_units=True)}


def main(_):
    # env_interface = EmbeddingInterfaceWrapper(TrainMarines())
    env_interface = EmbeddingInterfaceWrapper(BeaconEnvironmentInterface())
    learner = Learner(16, env_kwargs, env_interface, run_name="ImpBeacon1")
    learner.train()

    # Refresh environment every once in a while to deal with memory leak
    # environment = MultipleEnvironment(lambda: SCEnvironmentWrapper(env_interface, env_kwargs),
    #                                   num_instance=1)
    # agent = LSTMAgent(env_interface)
    # learner = ActorCriticLearner(environment, agent, run_name="SyncMarines", load_model=False)
    # i = 0
    # while True:
    #     i += 1
    #     print(i)
    #     learner.train_episode()

    # load_model = False
    # while True:
    #     num_instances = 1
    #     print("Starting environment")
    #     environment = MultipleEnvironment(lambda: SCEnvironmentWrapper(env_interface, env_kwargs),
    #                                       num_instance=num_instances)
    #     agent = LSTMAgent(env_interface)
    #     learner = ActorCriticLearner(environment, agent, run_name="SyncMarines", load_model=load_model)
    #     try:
    #         for i in range(1000):
    #             print("Starting environment2", i)
    #             learner.train_episode()
    #     finally:
    #         environment.close()
    #     load_model = True


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    # import sys
    # from absl import flags
    # FLAGS = flags.FLAGS
    # FLAGS(sys.argv)
    main(None)
