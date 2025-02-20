import torch
import torch.nn as nn
import numpy as np

from dreamer.modules.model import RSSM, RewardModel, ContinueModel, Discriminator
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic

from dreamer.utils.utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool
        self.use_classifier = config.parameters.dreamer.use_classifier

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if self.use_classifier:
            self.classifier = Discriminator(action_size, config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())
        if self.use_classifier:
            self.model_params += list(self.classifier.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def _log_gradients(self, model, prefix):
        """Log gradient norms for model parameters"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar(
                    f"gradients/{prefix}_{name}_norm",
                    param.grad.norm().item(),
                    self.num_total_episode
                )

    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        for iteration in range(self.config.train_iterations):
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(posteriors, deterministics)

            self.environment_interaction(env, self.config.num_interaction_episodes)
            self.evaluate(env)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        if self.use_classifier:
            classifier_loss = self._intrinsic_reward_loss(
                z=posterior_info.posteriors[:, :-1].reshape(-1, self.config.stochastic_size).detach(),
                action_batch=data.action[:,1:-1].reshape(-1, self.action_size).detach(),
                z_next=posterior_info.posteriors[:, 1:].reshape(-1, self.config.stochastic_size).detach(),
                z_next_prior=posterior_info.priors[:, 1:].reshape(-1, self.config.stochastic_size).detach(),
            )

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        if self.use_classifier:
            model_loss += classifier_loss

        # Log model losses before optimizer step
        self.writer.add_scalar("model/kl_loss", kl_divergence_loss.item(), self.num_total_episode)
        self.writer.add_scalar("model/reconstruction_loss", -reconstruction_observation_loss.mean().item(), self.num_total_episode)
        self.writer.add_scalar("model/reward_loss", -reward_loss.mean().item(), self.num_total_episode)
        self.writer.add_scalar("model/total_loss", model_loss.item(), self.num_total_episode)
        if self.config.use_continue_flag:
            self.writer.add_scalar("model/continue_loss", continue_loss.mean().item(), self.num_total_episode)
        if self.use_classifier:
            self.writer.add_scalar("model/classifier_loss", classifier_loss.item(), self.num_total_episode)

        self.model_optimizer.zero_grad()
        model_loss.backward()

        # Log model gradients
        self._log_gradients(self.encoder, "encoder")
        self._log_gradients(self.decoder, "decoder") 
        self._log_gradients(self.rssm, "rssm")
        self._log_gradients(self.reward_predictor, "reward")

        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()
    
    def _intrinsic_reward_loss(self, z, action_batch, z_next, z_next_prior):
        ip_batch_shape = z.shape[0]
        false_batch_idx = np.random.choice(ip_batch_shape, ip_batch_shape//2, replace=False)
        z_next_target = z_next 
        z_next_target[false_batch_idx] = z_next_prior[false_batch_idx]

        labels = torch.ones(ip_batch_shape, dtype=torch.long, device=self.device)
        labels[false_batch_idx] = 0.0

        logits = self.classifier(z, action_batch, z_next_target)
        classifier_loss = nn.CrossEntropyLoss()(logits, labels)

        return classifier_loss

    def behavior_learning(self, states, deterministics):
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # continue_predictor reinit
        # Unrolling. 
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic, actions=action
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors[:,:-1], behavior_learning_infos.deterministics[:,:-1]
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        if self.use_classifier:
            kl_rewards = self.classifier.get_reward(
                z=behavior_learning_infos.priors[:,:-1],
                a=behavior_learning_infos.actions[:,:-1],
                z_next=behavior_learning_infos.priors[:,1:]
            )
            predicted_rewards += (self.config.lambda_cost*kl_rewards)

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        # Log actor loss before optimizer step
        self.writer.add_scalar("agent/actor_loss", actor_loss.item(), self.num_total_episode)
        self.writer.add_scalar("agent/predicted_values", values.mean().item(), self.num_total_episode)
        self.writer.add_scalar("agent/lambda_values", lambda_values.mean().item(), self.num_total_episode)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Log actor gradients
        self._log_gradients(self.actor, "actor")

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        # Log critic loss before optimizer step
        self.writer.add_scalar("agent/critic_loss", value_loss.item(), self.num_total_episode)

        self.critic_optimizer.zero_grad()
        value_loss.backward()

        # Log critic gradients
        self._log_gradients(self.critic, "critic")

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        for epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = env.step(env_action)
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training/score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("evaluation/score", evaluate_score, self.num_total_episode)
