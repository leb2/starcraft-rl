from functools import partial
from absl import flags
from pysc2.lib import point_flag
from absl import app

from pysc2.env import sc2_env
import tensorflow as tf
import numpy as np
import trfl
import os

from pysc2.lib import actions

from multiprocessing import Process, Pipe, set_start_method
from threading import Thread
import time

from agent import LSTMAgent
from env_interface import EmbeddingInterfaceWrapper, BeaconEnvironmentInterface
from environment import SCEnvironmentWrapper


def run_actor(actor_factory):
    actor = actor_factory()
    while True:
        actor.get_params()
        actor.send_trajectory(actor.generate_trajectory())


def learn(learner, pipe):
    while True:
        # print("[Learner] waiting for message on pipe: ", learner.pipes.index(pipe))
        endpoint, data = pipe.recv()

        if endpoint == "get_params":
            with learner.agent.graph.as_default():
                # print("[Learner] received get params")
                trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                names = [var.name for var in trainable_variables]
                variables = learner.session.run(trainable_variables)
                pipe.send({name: variable for name, variable in zip(names, variables)})
                # print("[Learner] sent params")

        elif endpoint == "add_trajectory":
            # print("[Learner] received add trajectory")
            learner.add_trajectory(data)
        else:
            raise Exception("Invalid endpoint")


class Learner:
    def __init__(self, num_actors, load_model=False):
        self.num_actors = num_actors
        self.pipes = []
        self.processes = []
        self.threads = []
        self.trajectory_queue = []
        self.weights_dir = './weights'
        self.weights_path = os.path.join(self.weights_dir, 'model.ckpt')
        self.epoch = 0

        self.discount_factor = 0.9
        self.td_lambda = 0.9

        self.env_interface = EmbeddingInterfaceWrapper(BeaconEnvironmentInterface())
        self.agent = LSTMAgent(self.env_interface)

        with self.agent.graph.as_default():
            self.rewards_input = tf.placeholder(tf.float32, [None], name="rewards")  # T
            self.behavior_log_probs_input = tf.placeholder(tf.float32, [None, None], name="behavior_log_probs")  # T
            self.loss = self._ac_loss()
            # self.loss = self._impala_loss()
            self.train_op = tf.train.AdamOptimizer(0.0003).minimize(self.loss)
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if load_model:
                try:
                    self._load_model()
                except ValueError:
                    print("Could not load model")

    def start_children(self):
        for process_id in range(self.num_actors):
            parent_conn, child_conn = Pipe()
            self.pipes.append(parent_conn)
            p = Process(target=run_actor, args=(partial(Actor, child_conn, self.env_interface),))
            self.processes.append(p)
            p.start()

        for i in range(self.num_actors):
            t = Thread(target=learn, args=(self, self.pipes[i]))
            self.threads.append(t)
            t.start()

    def train(self):
        self.start_children()
        while True:
            # print("[Learner] Sleeping")
            time.sleep(0.01)
            if len(self.trajectory_queue) >= 1:
                self.update_model(self.trajectory_queue)
                self.trajectory_queue = []

    def add_trajectory(self, trajectory):
        self.update_model([trajectory])
        # self.trajectory_queue.append(trajectory)

    def update_model(self, rollouts):
        for i in range(len(rollouts)):
            rollout = rollouts[i]
            if rollout.done:
                feed_dict = {
                    self.rewards_input: rollout.rewards,
                    **self.agent.get_feed_dict(rollout.states, rollout.masks, rollout.actions, rollout.bootstrap_state)
                }

                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)

        self.epoch += 1
        if self.epoch % 50 == 0:
            self.save_model()
        with open('rewards.txt', 'a+') as f:
            for r in rollouts:
                f.write('%d\n' % r.total_reward())

    def save_model(self):
        """
        Saves the current model weights in current `save_path`.
        """
        save_path = self.saver.save(self.session, self.weights_path)
        print("Model Saved in %s" % save_path)

    def _load_model(self):
        """
        Loads the model from weights stored in the current `save_path`.
        """
        self.saver.restore(self.session, self.weights_path)
        print('Model Loaded')

    def _impala_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        rewards = tf.expand_dims(self.rewards_input, axis=1)

        all_values = self.agent.train_values()
        values = tf.expand_dims(all_values[:-1], axis=1)
        bootstrap = tf.expand_dims(all_values[-1], axis=0)
        train_log_probs = self.agent.train_log_probs()

        log_rhos = train_log_probs - self.behavior_log_probs_input
        vs, advantage = trfl.vtrace_from_importance_weights(log_rhos, discounts, rewards, values, bootstrap)

        loss_actor = tf.reduce_mean(-tf.stop_gradient(advantage) * train_log_probs)
        loss_critic = tf.reduce_mean((vs - values) ** 2)
        result = loss_actor + 0.5 * loss_critic
        return result

    def _ac_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        rewards = tf.expand_dims(self.rewards_input, axis=1)

        values = tf.expand_dims(self.agent.train_values(), axis=1)
        bootstrap = tf.expand_dims(self.agent.bootstrap_value(), axis=0)
        glr = trfl.generalized_lambda_returns(rewards, discounts, values, bootstrap, lambda_=self.td_lambda)
        advantage = tf.squeeze(glr - values)

        loss_actor = tf.reduce_mean(-tf.stop_gradient(advantage) * self.agent.train_log_probs())
        loss_critic = tf.reduce_mean(advantage ** 2)
        result = loss_actor + 0.5 * loss_critic
        return result


class Actor:
    def __init__(self, pipe, env_interface):
        env_kwargs = {
            "map_name": "MoveToBeacon",
            "visualize": False,
            "step_mul": 64,
            'game_steps_per_episode': None,
            "agent_interface_format": sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=84,
                    minimap=84),
                action_space=actions.ActionSpace.FEATURES,
                use_feature_units=True)}
        self.env_interface = env_interface
        self.agent = LSTMAgent(self.env_interface)
        with self.agent.graph.as_default():
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.variable_names = [v.name for v in tf.trainable_variables()]

            self.assign_placeholders = {t.name: tf.placeholder(t.dtype, t.shape)
                                        for t in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
            self.assigns = [tf.assign(tensor, self.assign_placeholders[tensor.name])
                            for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

        self.env_interface = env_interface
        self.env = SCEnvironmentWrapper(self.env_interface, env_kwargs=env_kwargs)
        # self.env = MultipleEnvironment(lambda: SCEnvironmentWrapper(self.env_interface, env_kwargs=env_kwargs),
        #                                   num_instance=1)

        self.curr_iteration = 0
        self.pipe = pipe

    def generate_trajectory(self):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Returns each trajectory in the form of rollouts.
        """
        states, masks, _, _ = self.env.reset()
        memory = None
        rollout = Rollout()

        while True:
            action_indices, memory = self.agent.step(states, masks, memory)
            new_states, new_masks, rewards, dones = self.env.step(action_indices)

            rollout.add_step(states[0], action_indices[0], rewards[0], masks[0], dones[0])
            states = new_states
            masks = new_masks
            if all(dones):
                # Add in the done state for rollouts which just finished for calculating the bootstrap value.
                rollout.add_step(states[0])
                break
        self.curr_iteration += 1
        print("=============== Reward on iteration %d is [%.1f]" % (self.curr_iteration, rollout.total_reward()))
        return rollout

    def get_params(self):
        # print("[ACTOR] requesting params")
        self.pipe.send(("get_params", None))
        # print("[ACTOR] waiting for params")
        names_to_params = self.pipe.recv()
        # print("[ACTOR] got params")

        # for key, value in names_to_params.items():
        #     print("KEY: ", key, ", ", value.shape)

        with self.agent.graph.as_default():
            # n = "actor_spatial_x/dense/kernel:0"
            self.session.run(self.assigns, feed_dict={
                self.assign_placeholders[name]: names_to_params[name] for name in self.variable_names
            })
            # for var in tf.
            # for var in tf.trainable_variables():
            #     print(var.name)
            #     if var.name == n:
            #         print("kernel sum is ", np.sum(self.session.run(var)))

        # print("[ACTOR] Finished updating params")

    def send_trajectory(self, trajectory):
        # print("[ACTOR] sending trajectory:")
        self.pipe.send(("add_trajectory", trajectory))


class Rollout:
    """ Contains data needed for training from a single trajectory of the environment.
    Attributes:
        states: List of numpy arrays of shape [*state_shape], representing every state at which an action was taken.
        actions: List of action indices generated by the agent's step function.
        rewards: List of scalar rewards, representing the reward recieved after performing the corresponding action at
            the corresponding state.
        masks: List of masks generated by the environment.
        bootstrap_state: A numpy array of shape [*state_shape]. Represents the terminal state in the trajectory and is
            used to bootstrap the advantage estimation.
    """
    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.masks = []
        self.should_bootstrap = None
        self.bootstrap_state = None
        self.done = False

    def total_reward(self):
        """
        :return: The current sum of rewards recieved in the trajectory.
        """
        return np.sum(self.rewards)

    def add_step(self, state, action=None, reward=None, mask=None, done=None):
        """ Saves a step generated by the agent to the rollout.
        Once `add_step` sees a `done`, it stops adding subsequent steps. However, make sure to call `add_step` at
        least one more time in order to record the terminal state for bootstrapping. Only leave the keyword parameters
        as None if feeding in the terminal state.
        :param state: The state which the action was taken in.
        :param action: The action index of the action taken, generated by the agent.
        :param reward: The reward recieved from the environment after taken the action.
        :param mask: The action mask that was used during the step.
        :param done: Whether the action resulted in the environment reaching a terminal state
        """
        if not self.done:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.masks.append(mask)
            self.done = done
        elif self.bootstrap_state is None:
            self.bootstrap_state = state


def main(_):
    learner = Learner(4, load_model=False)
    learner.train()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    FLAGS = flags.FLAGS
    app.run(main)


