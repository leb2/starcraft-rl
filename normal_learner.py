from multiprocessing import set_start_method

import trfl
from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions
import os
import tensorflow as tf

from agent import ConvAgent
from env_interface import EmbeddingInterfaceWrapper, BeaconEnvironmentInterface
from environment import SCEnvironmentWrapper
from impala_master import Rollout


class NormalActor:
    def __init__(self, env_interface, load_model=False):
        mineral_env_config = {
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
        self.agent = ConvAgent(self.env_interface)
        self.weights_dir = './weights'
        self.weights_path = os.path.join(self.weights_dir, 'model.ckpt')
        self.env_interface = env_interface
        self.env = SCEnvironmentWrapper(self.env_interface, env_kwargs=mineral_env_config)
        self.curr_iteration = 0
        self.epoch = 0
        self.discount_factor = 0.7
        self.td_lambda = 0.9

        with self.agent.graph.as_default():
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.rewards_input = tf.placeholder(tf.float32, [None], name="rewards")  # T
            self.behavior_log_probs_input = tf.placeholder(tf.float32, [None, None], name="behavior_log_probs")  # T
            self.saver = tf.train.Saver()
            self.loss = self._ac_loss()
            self.train_op = tf.train.AdamOptimizer(0.0003).minimize(self.loss)
            if load_model:
                try:
                    self._load_model()
                except ValueError:
                    print("Could not load model")

            self.variable_names = [v.name for v in tf.trainable_variables()]
            self.assign_placeholders = {t.name: tf.placeholder(t.dtype, t.shape)
                                        for t in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}
            self.assigns = [tf.assign(tensor, self.assign_placeholders[tensor.name])
                            for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            self.session.run(tf.global_variables_initializer())

    def generate_trajectory(self):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Returns each trajectory in the form of rollouts.
        """
        agent_states, agent_masks, _, dones = self.env.reset()
        rollout = Rollout()
        memory = self.agent.get_initial_memory(1)

        while not all(dones):
            agent_actions, next_memory, log_action_prob = self.agent.step(agent_states, agent_masks, memory)
            env_action_lists = self.env_interface.convert_actions(agent_actions)
            next_agent_states, next_masks, rewards, dones = self.env.step(env_action_lists)
            rollout.add_step(state=agent_states[0],
                             memory=memory[0],
                             mask=agent_masks[0],
                             action=agent_actions[0],
                             reward=rewards[0],
                             done=dones[0],
                             log_action_prob=log_action_prob[0]
                             )
            agent_states, agent_masks = next_agent_states, next_masks
            memory = next_memory

        rollout.add_step(state=agent_states[0], memory=memory[0])
        print("================== Iteration %d, reward: [%.1f]" % (self.curr_iteration, rollout.total_reward()))
        self.curr_iteration += 1
        return rollout

    def update_model(self, rollouts):
        for i in range(len(rollouts)):
            rollout = rollouts[i]
            if rollout.done:
                feed_dict = {
                    self.rewards_input: rollout.rewards,
                    self.behavior_log_probs_input: [rollout.log_action_probs],
                    **self.agent.get_feed_dict(rollout.states + [rollout.bootstrap_state],
                                               rollout.memories + [rollout.bootstrap_memory],
                                               rollout.masks,
                                               rollout.actions)
                }
                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)

        self.epoch += 1
        # print("[Learner] Finished update model, logging")
        if self.epoch % 50 == 0:
            self.save_model()
        with open('rewards.txt', 'a+') as f:
            for r in rollouts:
                f.write('%d\n' % r.total_reward())
        # print("[Learner] Done logging")

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

    def train(self):
        while True:
            self.update_model([self.generate_trajectory()])

    def _ac_loss(self):
        num_steps = tf.shape(self.rewards_input)[0]
        discounts = tf.ones((num_steps, 1)) * self.discount_factor
        rewards = tf.expand_dims(self.rewards_input, axis=1)

        all_values = self.agent.train_values()
        values = tf.expand_dims(all_values[:-1], axis=1)
        bootstrap = tf.expand_dims(all_values[-1], axis=0)
        glr = trfl.generalized_lambda_returns(rewards, discounts, values, bootstrap, lambda_=self.td_lambda)
        advantage = tf.squeeze(glr - values)

        loss_actor = tf.reduce_mean(-tf.stop_gradient(advantage) * self.agent.train_log_probs())
        loss_critic = tf.reduce_mean(advantage ** 2)
        result = loss_actor + 0.5 * loss_critic
        return result


def main(_):
    env_interface = EmbeddingInterfaceWrapper(BeaconEnvironmentInterface())
    learner = NormalActor(env_interface, load_model=False)
    learner.train()

    # learner = Learner(4, load_model=False)
    # learner.train()


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    FLAGS = flags.FLAGS
    app.run(main)
