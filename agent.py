from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from absl import flags
import matplotlib.pyplot as plt

import parts
import util


class ActorCriticAgent(ABC):
    """ Agent that can provide all of the necessary tensors to train with Actor Critic using `ActorCriticLearner`.
    Training is not batched over games, so the tensors only need to provide outputs for a single trajectory.
    """

    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.Session()
        self.graph = tf.get_default_graph()

    @abstractmethod
    def step(self, states, masks, memory):
        """
        Samples a batch of actions, given a batch of states and action masks.
        :param memory: Memory returned by the previous step, or None for the first step.
        :param masks: Tensor of shape [batch_size, num_actions]. Mask contains 1 for available actions and 0 for
            unavailable actions.
        :param states: Tensor of shape [batch_size, *state_size]
        :return:
            action_indices:
                A list of AgentActions. This will be passed back into self.get_feed_dict during training.
            memory:
                An arbitrary object that will be passed into step at the next timestep.
        """
        pass

    @abstractmethod
    def train_values(self):
        """
        :return: The tensor of shape [T] representing the estimated values of the states specified in self.get_feed_dict
        """
        pass

    @abstractmethod
    def train_log_probs(self):
        """
        :return:
            The tensor of shape [T] representing the log probability of performing the action in the state specified
            by the values in self.get_feed_dict
        """
        pass

    @abstractmethod
    def get_feed_dict(self, states, memory, masks, actions):
        """
        Get the feed dict with values for all placeholders that are dependenceies for the tensors
        `bootstrap_value`, `train_values`, and `train_log_probs`.
        :param memory: Memory corresponding to the state.
        :param masks: A numpy array of shape [T, num_actions].
        :param states: A numpy array of shape [T, *state_shape].
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate `train_values` and `train_log_probs`
        """

    def get_initial_memory(self, num_agents):
        return [None] * num_agents


class InterfaceAgent(ActorCriticAgent, ABC):
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.num_actions = self.interface.num_actions()
        self.num_spatial_actions = self.interface.num_spatial_actions()
        self.num_select_actions = self.interface.num_select_unit_actions()

        self.state_input = tf.placeholder(tf.float32, [None, *self.interface.state_shape],
                                          name='state_input')  # [batch, *state_shape]
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions()],
                                         name='mask_input')  # [batch, num_actions]

        self.action_input = tf.placeholder(tf.int32, [None], name='action_input')  # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2],
                                            name='spatial_input')  # [T, 2]   dimension size 2 for x and y
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")

    def get_feed_dict(self, states, memory, masks, actions=None):
        screens = [s['screen'] for s in states]
        feed_dict = {
            self.state_input: np.array(screens),
            self.mask_input: np.array(masks),
        }
        if actions is not None:
            nonspatial, spacial, _, _ = zip(*[a.as_tuple() for a in actions])
            spacial = [(-1, -1) if spacial is None else spacial for spacial in spacial]
            feed_dict[self.action_input] = np.array(nonspatial)
            feed_dict[self.spacial_input] = np.array(spacial)
        return feed_dict

    def _get_chosen_selection_probs(self, selection_probs, selection_choice):
        """
        :param selection_probs: Tensor of integers of shape [T, num_units, num_selection_actions]
        :param selection_choice: Tensor of shape [T] of type int
        :return:
        """
        selection_probs = util.index(selection_probs, selection_choice)  # [T, num_selection_actions]
        num_selection_actions = self.interface.num_select_unit_actions()

        index = (self.action_input - self.num_spatial_actions) % tf.convert_to_tensor(num_selection_actions)
        return util.index(selection_probs, index)  # [T]

    def _get_chosen_spacial_prob(self, spacial_probs, spacial_choice):
        spacial_probs = util.index(spacial_probs, spacial_choice)  # [T, num_screen_dimensions]
        return util.index(spacial_probs, self.action_input % tf.convert_to_tensor(self.num_spatial_actions))  # [T]

    def _train_log_probs(self, nonspatial_probs, spatial_probs=None, selection_probs=None):
        nonspatial_log_probs = tf.log(util.index(nonspatial_probs, self.action_input) + 1e-10)

        result = nonspatial_log_probs
        if spatial_probs is not None:
            probs_y = self._get_chosen_spacial_prob(spatial_probs[0], self.spacial_input[:, 1])
            probs_x = self._get_chosen_spacial_prob(spatial_probs[1], self.spacial_input[:, 0])
            spacial_log_probs = tf.log(probs_x + 1e-10) + tf.log(probs_y + 1e-10)
            result = result + tf.where(self.action_input < self.num_spatial_actions,
                                       x=spacial_log_probs,
                                       y=tf.zeros_like(spacial_log_probs))

        if selection_probs is not None:
            probs_selection = self._get_chosen_selection_probs(selection_probs, self.unit_selection_input)
            selection_log_prob = tf.log(probs_selection + 1e-10)
            is_select_action = tf.logical_and(self.action_input >= self.num_spatial_actions,
                                              self.action_input < self.num_spatial_actions + self.num_select_actions)
            result = result + tf.where(is_select_action,
                                       x=selection_log_prob,
                                       y=tf.zeros_like(selection_log_prob))
        return result

    def _probs_from_features(self, features):
        num_steps = tf.shape(self.mask_input)[0]
        nonspatial_probs = parts.actor_nonspatial_head(features[:num_steps], self.mask_input, self.num_actions)
        spatial_probs = parts.actor_spatial_head(features[:num_steps], screen_dim=84,
                                                 num_spatial_actions=self.num_spatial_actions)
        return nonspatial_probs, spatial_probs


class ConvAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.features = parts.conv_body(self.state_input)
        self.nonspatial_probs, self.spatial_probs = self._probs_from_features(self.features)

    def log_prob_numpy(self, actions, nonspatial_probs, spatial_probs):
        """
        :param actions: [EnvAction]
        :param spatial_probs: [2, T, 84, num_spatial_actions]
        :param nonspatial_probs: [T, num_actions]
        :return: [T]
        """
        log_probs = []
        for i, action in enumerate(actions):
            log_prob = np.log(nonspatial_probs[i, action.index])
            if action.index < self.interface.num_spatial_actions():  # Is a spatial action
                x, y = action.spatial_coords
                log_prob += np.log(spatial_probs[0, i, x, action.index]) + np.log(spatial_probs[1, i, y, action.index])
            log_probs.append(log_prob)
        return log_probs

    def step(self, state, mask, memory):
        screens = [s['screen'] for s in state]
        nonspatial_probs, spatial_probs = self.session.run(
            [self.nonspatial_probs, self.spatial_probs], {
                self.state_input: screens,
                self.mask_input: mask
            })
        sampled_actions = util.sample_action(self.interface, nonspatial_probs, spatial_probs)

        # TODO: This extra call can probably be removed somehow
        # train_log_probs = self.session.run([self.train_log_probs()], self.get_feed_dict(
        #     state, memory, mask, actions=sampled_actions
        # ))

        train_log_probs = self.log_prob_numpy(sampled_actions, nonspatial_probs, spatial_probs)
        x = spatial_probs[0][0]
        y = spatial_probs[1][0]
        grid = np.matmul(x, y.T)
        return sampled_actions, self.get_initial_memory(len(screens)), train_log_probs
        # return sampled_actions, self.get_initial_memory(len(screens)), np.zeros((len(state),))  # train_log_probs

    def train_log_probs(self):
        return self._train_log_probs(self.nonspatial_probs, spatial_probs=self.spatial_probs)

    def train_values(self):
        return parts.value_head(self.features)
