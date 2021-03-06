from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

from env_interface import ParamType
import util
import parts
from attention import SelfAttention
use_lstm = False
np.set_printoptions(precision=2, suppress=True)


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
                A list of indices that represents the chosen action at the state. This can be of any data type
                and will be passed back into self.get_feed_dict during training.
            memory:
                An arbitrary object that will be passed into step at the next timestep.
        """
        pass

    @abstractmethod
    def bootstrap_value(self):
        """
        :return: A scalar tensor representing the bootstrap value.
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
    def get_feed_dict(self, states, masks, actions, bootstrap_state):
        """
        Get the feed dict with values for all placeholders that are dependenceies for the tensors
        `bootstrap_value`, `train_values`, and `train_log_probs`.

        :param bootstrap_state: A numpy a array of shape [*state_shape] representing the terminal state.
        :param masks: A numpy array of shape [T, num_actions].
        :param states: A numpy array of shape [T, *state_shape].
        :param actions: A list of action indices with length T.
        :return: The feed dict required to evaluate `train_values` and `train_log_probs`
        """


class InterfaceAgent(ActorCriticAgent, ABC):
    def __init__(self, interface):
        super().__init__()
        self.interface = interface
        self.num_actions = self.interface.num_actions
        self.num_screen_dims = int(len(self.interface.screen_dimensions) / 2)

        self.state_input = tf.placeholder(tf.float32, [None, *self.interface.state_shape])  # [batch, *state_shape]
        self.other_features_input = tf.placeholder(tf.float32, [None, 1 + self.interface.num_actions])
        self.mask_input = tf.placeholder(tf.float32, [None, self.interface.num_actions])  # [batch, num_actions]
        self.action_input = tf.placeholder(tf.int32, [None])  # [T]
        self.spacial_input = tf.placeholder(tf.int32, [None, 2])  # [T, 2]   dimension size 2 for x and y

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        feed_dict = {
            self.state_input: np.array(states),
            self.mask_input: np.array(masks),
        }
        if actions is not None:
            nonspacial, spacial = zip(*actions)
            spacial = [(13, 27) if spacial is None else spacial for spacial in spacial]
            feed_dict[self.action_input] = np.array(nonspacial)
            feed_dict[self.spacial_input] = np.array(spacial)
        return feed_dict

    def sample_action_index(self, nonspacial_probs, spacial_probs_x, spacial_probs_y):
        """
        :return: Generates a list of  action index tuples of type (nonspacial_index, (spacial_x, spacial_y))
        """
        chosen_nonspacials = util.sample_multiple(nonspacial_probs)  # [num_games]
        action_indices = []
        for i, chosen_nonspacial in enumerate(chosen_nonspacials):
            if chosen_nonspacial < self.num_screen_dims:
                x = np.random.choice(self.interface.screen_dimensions[chosen_nonspacial],
                                     p=spacial_probs_x[chosen_nonspacial][i])
                y = np.random.choice(self.interface.screen_dimensions[chosen_nonspacial],
                                     p=spacial_probs_y[chosen_nonspacial][i])
                action_indices.append((chosen_nonspacial, (x, y)))
            else:
                action_indices.append((chosen_nonspacial, None))
        return action_indices

    def _train_log_probs(self, nonspacial_probs, spacial_probs_x, spacial_probs_y):
        nonspacial_log_probs = tf.log(util.index(nonspacial_probs, self.action_input) + 0.00000001)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(spacial_probs_x, self.spacial_input[:, 0])
        spacial_log_probs = tf.log(probs_x + 0.0000001) + tf.log(probs_y + 0.0000001)
        result = nonspacial_log_probs + tf.where(self.action_input < self.num_screen_dims,
                                                 x=spacial_log_probs,
                                                 y=tf.zeros_like(spacial_log_probs))
        return result

    def _get_chosen_spacial_prob(self, spacial_probs, spacial_choice):
        spacial_probs = tf.stack(spacial_probs, axis=-1)  # [T, screen_dim, num_screen_dimensions]
        spacial_probs = util.index(spacial_probs, spacial_choice)  # [T, num_screen_dimensions]
        return util.index(spacial_probs, self.action_input % tf.convert_to_tensor(self.num_screen_dims))  # [T]

    def _probs_from_features(self, features):
        nonspacial_probs = parts.actor_nonspacial_head(features, self.mask_input, self.num_actions)
        spacial_probs_x, spacial_probs_y = \
            parts.actor_spacial_head(features, self.interface.screen_dimensions)
        return nonspacial_probs, spacial_probs_x, spacial_probs_y


class ConvAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.bootstrap_state_input = tf.placeholder(tf.float32, [*self.interface.state_shape])

        self.features = parts.conv_body(self.state_input)
        self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y = self._probs_from_features(self.features)

    def step(self, state, mask, memory):
        probs = self.session.run(
            [self.nonspacial_probs, *self.spacial_probs_x, *self.spacial_probs_y], {
                self.state_input: state,
                self.mask_input: mask
            })
        nonspacial_probs = probs[0]
        spacial_probs = probs[1:]
        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]  # [num_screen_dims, num_games, screen_dim]

        return self.sample_action_index(nonspacial_probs, spacial_probs_x, spacial_probs_y), None

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        feed_dict = super(ConvAgent, self).get_feed_dict(states, masks, actions, bootstrap_state)
        return {
            self.bootstrap_state_input: np.array(bootstrap_state),
            **feed_dict
        }

    def train_log_probs(self):
        return self._train_log_probs(self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y)

    def bootstrap_value(self):
        return tf.squeeze(parts.value_head(parts.conv_body(tf.expand_dims(self.bootstrap_state_input, axis=0))), axis=0)

    def train_values(self):
        return parts.value_head(self.features)


class LSTMAgent(InterfaceAgent):
    def __init__(self, interface):
        super().__init__(interface)
        self.num_select_actions = self.interface.num_unit_selection_actions

        self.rnn_size = 256
        self.self_attention = SelfAttention(hidden_size=128, num_heads=2, attention_dropout=0, train=True)
        self.lstm = tf.contrib.rnn.LSTMCell(self.rnn_size)

        self.memory_input = tf.placeholder(tf.float32, [2, None, self.rnn_size], name="memory_input")
        self.unit_embeddings_input = tf.placeholder(tf.float32, [None, None, self.interface.unit_embedding_size],
                                                    name="unit_embeddings_input")
        self.unit_selection_input = tf.placeholder(tf.int32, [None], name="unit_selection_input")

        self.features = self.features()  # Shape [batch_size, num_features]

        lstm_output, self.next_lstm_state = self._lstm_step()
        self.train_output = self._lstm_step_train()
        self.all_values = parts.value_head(self.train_output)

        self.nonspacial_probs, self.spacial_probs_x, self.spacial_probs_y = self._probs_from_features(lstm_output)
        self.nonspacial_train, self.spacial_train_x, self.spacial_train_y = \
            self._probs_from_features(self.train_output[:-1])
        self.unit_selection_probs = self._selection_probs_from_features(lstm_output, self.unit_embeddings_input)
        self._f1 = lstm_output
        self.unit_selection_probs_train = self._selection_probs_from_features(self.train_output[:-1],
                                                                              self.unit_embeddings_input[:-1])

    def _selection_probs_from_features(self, features, embeddings):
        return parts.actor_pointer_head(features, embeddings, self.num_select_actions)

    def features(self):
        conv_features = parts.conv_body(self.state_input)
        unit_features = tf.reduce_sum(self.self_attention(self.unit_embeddings_input, bias=0), axis=1)
        logits = tf.concat([conv_features, unit_features, self.other_features_input], axis=1)
        logits = tf.layers.dense(logits, units=200, activation=tf.nn.leaky_relu, name='feature_layer')
        logits = tf.layers.dense(logits, units=200, activation=tf.nn.tanh, name='feature_layer2')
        return logits

    def step(self, states, masks, memories):
        """
        :param states: List of states of length batch size. In this case, state is a dict with keys:
            "unit_embeddings": numpy array with shape [num_units, embedding_size]
            "state": numpy array with shape [*state_shape]
        :param masks: numpy array of shape [batch_size, num_actions]
        :param memories: Array of memory dictionaries
        :return used_states is the state that was actually used, including modifications for memory
        """
        if memories is None:
            memories = []
            for _ in range(len(states)):
                memories.append({
                    'next_lstm_state': np.zeros((2, len(states), self.rnn_size)),
                    'prev_action': np.zeros(self.interface.num_actions)
                })
        for state, memory in zip(states, memories):
            state['prev_action'] = memory['prev_action']

        if use_lstm:
            feed_dict = {
                **self.get_feed_dict(states, masks),
                self.memory_input: memory['next_lstm_state']
            }
        else:
            feed_dict = {
                **self.get_feed_dict(states, masks),
            }
        results = self.session.run(
            [self.next_lstm_state, self.nonspacial_probs, self.unit_selection_probs,
             *self.spacial_probs_x, *self.spacial_probs_y], feed_dict)
        next_lstm_state, nonspacial_probs, selection_probs = results[:3]
        spacial_probs = results[3:]

        spacial_probs_x = spacial_probs[:self.num_screen_dims]
        spacial_probs_y = spacial_probs[self.num_screen_dims:]

        unit_coords = util.pad_stack([state['unit_coords'][:, :2] for state in states], pad_axis=0, stack_axis=0)
        sampled_action = self.sample_action_index_with_units(nonspacial_probs, spacial_probs_x,
                                                     spacial_probs_y, selection_probs, unit_coords)
        next_memories = []
        # TODO, to reimplement memory, you will probably have to np stack since memory is now a list of dictionaries
        # with a separate state for each item in the batch.
        for i in range(len(states)):
            next_memory = {'next_lstm_state': next_lstm_state, 'prev_action': np.zeros(self.interface.num_actions)}
            next_memory['prev_action'][sampled_action[i][0]] = 1
            next_memories.append(next_memory)
        return states, sampled_action, next_memories

    def _lstm_step(self):
        if use_lstm:
            state_tuple = tf.unstack(self.memory_input, axis=0)
            flattened_state = self.features
            lstm_output, next_state_tuple = self.lstm(flattened_state, state=state_tuple)
            next_state = tf.stack(next_state_tuple, axis=0)
            return lstm_output, next_state
        else:
            return self.features, tf.zeros(0)
            # memory = tf.zeros([2, 1, self.rnn_size], tf.float32)
            # state_tuple = tf.unstack(memory, axis=0)
            # return self.lstm(self.features, state=state_tuple)[0], tf.zeros(0)

    def _lstm_step_train(self):
        if use_lstm:
            # Removed the lstm stuff temporarily
            flattened_state = tf.expand_dims(self.features, axis=0)
            train_output, _ = tf.nn.dynamic_rnn(self.lstm, flattened_state, dtype=tf.float32)
            return tf.squeeze(train_output, axis=0)
        else:
            # memory = tf.zeros([2, 1, self.rnn_size], tf.float32)
            # state_tuple = tf.unstack(memory, axis=0)
            # return self.lstm(self.features, state=state_tuple)[0]
            return self.features

    def get_feed_dict(self, states, masks, actions=None, bootstrap_state=None):
        screens = np.stack([state['screen'] for state in states], axis=0)

        feed_dict = {
            self.mask_input: np.array(masks),
        }
        all_states = states if bootstrap_state is None else [*states, bootstrap_state]
        unit_embeddings = util.pad_stack([state['unit_embeddings'] for state in all_states], pad_axis=0, stack_axis=0)

        other_features = np.stack([state['other_features'] for state in all_states], axis=0)
        # for state in all_states:
            # print("This is the state")
            # print(state)
            # print("This is the prev_action")
            # print(state['prev_action'])
            # if len(masks) == 1:
            #     print("this is the mask: ")
            #     print(masks)
            # print(masks)
            # print()
        prev_action_onehot = np.stack([state['prev_action'] for state in all_states], axis=0)

        all_other_features = np.concatenate([other_features, prev_action_onehot], axis=-1)
        feed_dict[self.other_features_input] = all_other_features
        feed_dict[self.unit_embeddings_input] = unit_embeddings

        if bootstrap_state is not None:
            bootstrap_screen = np.expand_dims(bootstrap_state['screen'], axis=0)
            feed_dict[self.state_input] = np.concatenate([screens, bootstrap_screen], axis=0)
        else:
            feed_dict[self.state_input] = screens

        if actions is not None:
            nonspacial, spacials, selection_coords, selection_indices = zip(*actions)
            spacials = [(13, 27) if spacial is None else spacial for spacial in spacials]
            selections = [-1 if selection is None else selection for selection in selection_indices]
            feed_dict[self.action_input] = np.array(nonspacial)
            feed_dict[self.spacial_input] = np.array(spacials)
            feed_dict[self.unit_selection_input] = np.array(selections)
        return feed_dict

    def bootstrap_value(self):
        return self.all_values[-1]

    def train_values(self):
        return self.all_values[:-1]

    def train_log_probs(self):
        return self._train_log_probs_with_units(self.nonspacial_train, self.spacial_train_x, self.spacial_train_y,
                                                self.unit_selection_probs_train)

    def sample_action_index_with_units(self, nonspacial_probs, spacial_probs_x, spacial_probs_y,
                                       unit_distribution, unit_coords):
        """
        unit_distribution is an array of shape [num_games, num_select_actions, num_units]
        :return: Generates an list of action index tuple of type (nonspacial_index, (spacial_x, spacial_y))
        """
        # print("nonspatial probs are")
        # print(nonspacial_probs)

        actions = self.sample_action_index(nonspacial_probs, spacial_probs_x, spacial_probs_y)
        num_units = unit_distribution.shape[2]
        new_actions = []
        unit_coords = np.stack(unit_coords)

        for i, action in enumerate(actions):
            nonspacial_index, coords = action
            param_type, param_index = self.interface.action_parameter_type(nonspacial_index)
            if param_type is ParamType.SELECT_UNIT:
                unit_choice = np.random.choice(num_units, p=unit_distribution[i, param_index])
                new_actions.append((nonspacial_index, coords,
                                    unit_coords[i, unit_choice].astype(np.int32), unit_choice))
            else:
                new_actions.append((nonspacial_index, coords, None, None))
        return new_actions

    def _get_chosen_selection_probs(self, selection_probs, selection_choice):
        """
        :param selection_probs: Tensor of integers of shape [T, num_units, num_selection_actions]
        :param selection_choice: Tensor of shape [T] of type int
        :return:
        """
        selection_probs = util.index(selection_probs, selection_choice)  # [T, num_selection_actions]
        num_selection_actions = self.interface.num_unit_selection_actions

        index = (self.action_input - self.num_screen_dims) % tf.convert_to_tensor(num_selection_actions)
        return util.index(selection_probs, index)  # [T]

    def _train_log_probs_with_units(self, nonspacial_probs, spacial_probs_x, spacial_probs_y, selection_probs):
        nonspacial_log_probs = tf.log(util.index(nonspacial_probs, self.action_input) + 1e-10)

        # TODO: This only works if all screen dimensions are the same. Should pad to greatest length
        probs_y = self._get_chosen_spacial_prob(spacial_probs_y, self.spacial_input[:, 1])
        probs_x = self._get_chosen_spacial_prob(spacial_probs_x, self.spacial_input[:, 0])
        probs_selection = self._get_chosen_selection_probs(selection_probs, self.unit_selection_input)

        selection_log_prob = tf.log(probs_selection + 1e-10)
        spacial_log_probs = tf.log(probs_x + 1e-10) + tf.log(probs_y + 1e-10)

        result = nonspacial_log_probs
        result = result + tf.where(self.action_input < self.num_screen_dims,
                                   x=spacial_log_probs,
                                   y=tf.zeros_like(spacial_log_probs))

        is_select_action = tf.logical_and(self.action_input >= self.num_screen_dims,
                                          self.action_input < self.num_screen_dims + self.num_select_actions)
        result = result + tf.where(is_select_action,
                                   x=selection_log_prob,
                                   y=tf.zeros_like(selection_log_prob))
        return result