from absl import app
from absl import flags
import os
import pathlib

from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.maps import lib

import env_interface as interfaces
from agent import LSTMAgent
from environment import MultipleEnvironment, SCEnvironmentWrapper
from normal_learner import ActorCriticLearner

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("render", True, "Whether to render with pygame.")
    point_flag.DEFINE_point("feature_screen_size", "84",
                            "Resolution for screen feature layers.")
    point_flag.DEFINE_point("feature_minimap_size", "64",
                            "Resolution for minimap feature layers.")
    point_flag.DEFINE_point("rgb_screen_size", None,
                            "Resolution for rendered screen.")
    point_flag.DEFINE_point("rgb_minimap_size", None,
                            "Resolution for rendered minimap.")
    flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                      "Which action space to use. Needed if you take both feature "
                      "and rgb observations.")
    flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")
    flags.DEFINE_bool("epsilon", False, "Whether to use epsilon greedy")

    flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
    flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
    flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
    flags.DEFINE_integer("step_mul", 24, "Game steps per agent step.")

    flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                      "Agent 1's race.")
    flags.DEFINE_bool("cuda", True, "Whether to train on gpu")
    flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
    flags.DEFINE_integer("save_replay_episodes", 0, "How many episodes per replay save, 0 for no saves.")

    # flags.DEFINE_string("map", None, "Name of a map to use.")
    flags.DEFINE_string("save_name", "recent", "Save run information under ./saves/<save_name>")

    flags.DEFINE_bool("load_model", False, "Whether to load the previous run's model")

    flags.DEFINE_float("gamma", 0.96, "Discount factor")
    flags.DEFINE_float("learning_rate", 0.0003, "Learning rate")
    flags.DEFINE_float("td_lambda", 0.96, "Lambda value for generalized advantage estimation")


def main(unused_argv):
    map_name = "MoveToBeacon"
    save_name = "Beacon5"
    load_run_name = None
    num_instances = 8
    step_mul = 8
    load_body_only = False

    save_dir = os.path.join('saves', save_name)
    env_kwargs = {
        'map_name': map_name,
        'players': [sc2_env.Agent(sc2_env.Race[FLAGS.agent_race])],
        'agent_interface_format': sc2_env.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=FLAGS.rgb_screen_size,
            rgb_minimap=FLAGS.rgb_minimap_size,
            action_space=FLAGS.action_space,
            use_feature_units=True),
        'step_mul': step_mul,
        'game_steps_per_episode': FLAGS.game_steps_per_episode,
        'disable_fog': FLAGS.disable_fog,
        'visualize': False,
        'save_replay_episodes': 0,
        'replay_dir': os.path.join(save_dir, 'replays')
    }

    if map_name in {'DefeatRoaches', 'StalkersVsRoaches'}:
        interface = interfaces.EmbeddingInterfaceWrapper(interfaces.RoachesEnvironmentInterface())
    elif map_name == 'MoveToBeacon':
        interface = interfaces.EmbeddingInterfaceWrapper(interfaces.BeaconEnvironmentInterface())
    elif map_name in {'BuildMarines', 'EconomicRLTraining'}:
        interface = interfaces.EmbeddingInterfaceWrapper(interfaces.TrainMarines())
    else:
        raise Exception('Unsupported Map')

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    run_info_path = os.path.join(save_dir, 'info.txt')
    with open(run_info_path, 'a') as f:
        for key in FLAGS.__flags:
            f.write("%s, %s\n" % (key, FLAGS.__flags[key]._value))
        f.write('\n\n\n')

    # Refresh environment every once in a while to deal with memory leak
    for j in range(1):
        environment = MultipleEnvironment(lambda: SCEnvironmentWrapper(interface, env_kwargs),
                                          num_instance=num_instances)
        agent = LSTMAgent(interface)
        learner = ActorCriticLearner(environment, agent,
                                     run_name=save_name,
                                     load_run_name=load_run_name,
                                     gamma=0.95,
                                     td_lambda=0.95,
                                     learning_rate=0.0001,
                                     load_body_only=load_body_only)
        load_body_only = False
        try:
            for i in range(500):
                if FLAGS.max_episodes and i >= FLAGS.max_episodes:
                    break
                learner.train_episode()
        finally:
            environment.close()


if __name__ == "__main__":
    app.run(main)