import vtrace as trfl
import os
import tensorflow as tf
from impala_master import Rollout


class ActorCriticLearner:
    """ Implementation of generalized advantage actor critic for TensorFlow.
    """
    def __init__(self, environment, agent,
                 run_name="temp",
                 load_run_name=None,
                 load_body_only=False,
                 gamma=0.96,
                 td_lambda=0.96,
                 learning_rate=0.0003):
        """
        :param environment: An instance of `MultipleEnvironment` to be used to generate trajectories.
        :param agent: An instance of `ActorCriticAgent` to be used to generate actions.
        :param run_name: The directory to store rewards and weights in.
        :param load_model: True if the model should be loaded from `save_dir`.
        :param gamma: The discount factor.
        :param td_lambda: The value of lambda used in generalized advantage estimation. Set to 1 to behave like
            monte carlo returns.
        """
        self.env = environment
        self.num_games = self.env.num_instances
        self.agent = agent
        self.discount_factor = gamma
        self.td_lambda = td_lambda

        project_root = os.path.dirname(os.path.realpath(__file__))
        self.save_dir = os.path.join(project_root, 'saves', run_name)
        self.code_dir = os.path.join(self.save_dir, 'code')
        self.weights_dir = os.path.join(self.save_dir, 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'model.ckpt')

        if load_run_name is None:
            self.load_weights_path = self.weights_path
        else:
            self.load_weights_path = os.path.join(project_root, 'saves', load_run_name, 'weights', 'model.ckpt')

        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        if not os.path.exists(self.code_dir):
            os.makedirs(self.code_dir)

        os.system('cp -r ' + os.path.join(project_root, './*.py') + ' ' + self.code_dir)
        self.rewards_path = os.path.join(self.save_dir, 'rewards.txt')
        self.episode_counter = 0

        self.rollouts = [Rollout() for _ in range(self.num_games)]
        with self.agent.graph.as_default():
            self.rewards_input = tf.placeholder(tf.float32, [None])
            self.loss = self._ac_loss()
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.session = self.agent.session
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            try:
                self.load_model(self.load_weights_path)
                print("Could load model1")
            except ValueError:
                print("Could not load model2")
            if load_body_only:
                self.reinitialize_head()

    def reinitialize_head(self):
        print("Reinitializing head")
        body_keywords = ["pointer_head/dense/", "pointer_head/dense_1/", "shared", "lstm"]
        head_variables = [v for v in tf.trainable_variables() if not any([keyword in v.name for keyword in body_keywords])]
        print("head variables are")
        for var in head_variables:
            print(var)

        body_variables = [v for v in tf.trainable_variables() if any([keyword in v.name for keyword in body_keywords])]
        print("Body variables are")
        for var in body_variables:
            print(var)
        self.session.run(tf.initialize_variables(head_variables))

    def train_episode(self):
        """ Trains the agent for single episode for each environment in the `MultipleEnvironment`.
        Training is synchronized such that all training happens after all agents have finished acting in the
        environment. Call this method in a loop to train the agent.
        """
        self.generate_trajectory()
        for i in range(self.num_games):
            rollout = self.rollouts[i]
            if rollout.done:
                feed_dict = {
                    self.rewards_input: rollout.rewards,
                    **self.agent.get_feed_dict(rollout.states, rollout.masks, rollout.actions, rollout.bootstrap_state)
                }

                loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
                self._log_data(rollout.total_reward())
                self.rollouts[i] = Rollout()

    def generate_trajectory(self):
        """
        Repeatedly generates actions from the agent and steps in the environment until all environments have reached a
        terminal state. Stores the complete result from each trajectory in `rollouts`.
        """
        states, masks, _, _ = self.env.reset()
        memory = None
        while True:
            action_indices, memory, log_probs = self.agent.step(states, masks, memory)
            new_states, new_masks, rewards, dones = self.env.step(action_indices)

            for i, rollout in enumerate(self.rollouts):
                rollout.add_step(states[i], action_indices[i], rewards[i], masks[i], dones[i])
            states = new_states
            masks = new_masks
            if all(dones):
                # Add in the done state for rollouts which just finished for calculating the bootstrap value.
                for i, rollout in enumerate(self.rollouts):
                    rollout.add_step(states[i])
                return

    def save_model(self):
        """
        Saves the current model weights in current `save_path`.
        """
        save_path = self.saver.save(self.session, self.weights_path)
        print("Model Saved in %s" % save_path)

    def load_model(self, path):
        """
        Loads the model from weights stored in the current `save_path`.
        """
        self.saver.restore(self.session, path)
        print('Model Loaded')

    def _log_data(self, reward):
        self.episode_counter += 1
        with open(self.rewards_path, 'a+') as f:
            f.write('%d\n' % reward)

        if self.episode_counter % 50 == 0:
            self.save_model()

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
