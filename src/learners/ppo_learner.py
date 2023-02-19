import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.ppo import PPOCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from .__optimizers__ import Optimizer

"""
? PPO general questions:
2. Why we use G_t for the critic rather than TD estimation?
$3. Do I need to calculate the ratio between adjacent episodes?

? OPTIONS:
* Increased or time-dependent entropy
* Calibrate clip coefficient
* RNN critic?
! Annealing LR; clipping V-loss
"""


class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = PPOCritic(scheme, args)
        if self.args.critic_td:
            self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        #! Consider using ADAM optimizer with eps=1e-5
        # self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.agent_optimiser = Optimizer(args.optimizer, self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps).__get__()
        self.critic_optimiser = Optimizer(args.optimizer, self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps).__get__()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if (episode_num + 1) % batch.batch_size != 0:
            return

        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = (mask[:, 1:] * (1 - terminated[:, :-1]))
        avail_actions = batch["avail_actions"][:, :-1]

        critic_log = {key:[] for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "value_mean", "target_mean"]}
        actor_log  = {key:[] for key in ["advantage_mean", "ppo_loss", "agent_grad_norm", "pi_max"]}
        critic_mask = mask.clone()

        # mask = mask.repeat(1, 1, self.n_agents).view(-1) #!!! ????
        mask = mask.squeeze()

        targets = self._calc_targets(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)
        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        base_pi_taken = th.gather(mac_out, dim=-1, index=actions).squeeze()
        base_pi_taken[mask == 0] = 1.0 # to prevent -inf
        log_base_pi_taken = th.log(base_pi_taken).detach()

        for epoch in range(self.args.epochs):
            # Calculate updated logits
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Mask out unavailable actions, renormalise (as in action selection)
            mac_out[avail_actions == 0] = 0
            mac_out = mac_out/(mac_out.sum(dim=-1, keepdim=True) + 1e-5)

            pi_taken = th.gather(mac_out, dim=-1, index=actions).squeeze()
            pi_taken[mask == 0] = 1.0 # to prevent -inf
            log_pi_taken = th.log(pi_taken)

            # Calculate the critic loss & optimize step for the critic
            values, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t, targets=targets)
            for key in critic_log.keys():
                critic_log[key].append(critic_train_stats[key])

            # Advantage for surrotage loss
            advantages = (targets[:, :-1] - values[:, :-1]).squeeze(-1).detach()
            if self.args.adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Calculate entropy to improve exploration
            entropy = (-mac_out * th.log2(mac_out + 1e-8)).sum(axis=-1).mean()


            # Calculate the actor (PPO) loss
            ratio = (log_pi_taken - log_base_pi_taken).exp()
            surr_loss1 = -advantages * ratio
            surr_loss2 = -advantages * th.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
            ppo_loss = th.max(surr_loss1, surr_loss2).mean() - self.args.ent_coefficient * entropy
            
            # Optimise agents
            self.agent_optimiser.zero_grad()
            ppo_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()
        
            actor_log["advantage_mean"].append((advantages * mask).sum().item() / mask.sum().item())
            actor_log["ppo_loss"].append(ppo_loss.item())
            actor_log["agent_grad_norm"].append(grad_norm)
            actor_log["pi_max"].append(((mac_out.max(dim=-1)[0].squeeze() * mask).sum().item() / mask.sum().item()))

        # Applied only in "TD-Critic" mode
        if self.args.critic_td and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        #$ I'm here but need to run the previous code for testing $# 

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_log["critic_loss"])
            for key in critic_log.keys():
                self.logger.log_stat(key, sum(critic_log[key])/ts_logged, t_env)

            ts_logged = len(actor_log["advantage_mean"])
            for key in actor_log.keys():
                self.logger.log_stat(key, sum(actor_log[key])/ts_logged, t_env)

            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t, targets):
        #! 1. Updateing the critic to be RNN rather than DNN requires some corrections in this function (run seq, not paralel)
        self.critic_training_steps += 1
        values = self.critic(batch)[:, :].squeeze(-1)

        # 0-out the targets that came from padded data
        td_error = (values - targets)
        masked_td_error = td_error[:, :-1] * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log = {}
        running_log["critic_loss"] = loss.item()
        running_log["critic_grad_norm"]= grad_norm
        mask_elems = mask.sum().item()
        running_log["td_error_abs"] = (masked_td_error.abs().sum().item() / mask_elems)
        running_log["value_mean"]= (values[:, :-1] * mask).sum().item() / mask_elems
        running_log["target_mean"]= (targets[:, :-1] * mask).sum().item() / mask_elems

        return values, running_log

    def _calc_targets(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Calculate target values based on the observation, relevant for all the epochs
        values = self.target_critic(batch)[:, :].squeeze(-1) if self.args.critic_td \
            else self.critic(batch)[:, :].squeeze(-1)

        targets = th.zeros_like(values)
        targets[:, -1] = (1 - terminated.sum(axis=1)) * values[:, -1]

        # Calculate td-lambda targets
        if self.args.critic_td:
            targets[:, :-1] = self.args.gamma * values[:, 1:] + rewards
        else:
            for t in reversed(range(rewards.size(1))):
                targets[:, t] = targets[:, t+1] * self.args.gamma + rewards[:, t]
        
        return targets.detach()

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
