from multiprocessing import set_start_method

from pysc2.env import sc2_env
from pysc2.lib import actions

from env_interface import EmbeddingInterfaceWrapper, TrainMarines, BeaconEnvironmentInterface
from impala_master import Learner


env_kwargs = {
    # "map_name": "EconomicRLTraining",
    # "map_name": "StalkersVsRoaches",
    "map_name": "MoveToBeacon",
    "visualize": False,
    "step_mul": 8,
    'game_steps_per_episode': None,
    "agent_interface_format": sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
            screen=84,
            minimap=84),
        action_space=actions.ActionSpace.FEATURES,
        use_feature_units=True)}


def main(_):
    env_interface = EmbeddingInterfaceWrapper(BeaconEnvironmentInterface())
    # env_interface = EmbeddingInterfaceWrapper(TrainMarines())
    learner = Learner(10, env_kwargs, env_interface, run_name="TestRefactor", load_model=True)
    learner.train()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main(None)
