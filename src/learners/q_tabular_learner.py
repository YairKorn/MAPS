from enum import unique
import torch as th
import numpy as np
from .tabular_learner import TabularLearner
from components.episode_buffer import EpisodeBatch

class QTabularLearner(TabularLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.buffer = self.mac.action_model.buffer


    """ A learner of MAPS architecture """
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        observations = batch["obs"][:, :-1]
        rewards = self.buffer["reward"][:, :(self.mac.n_agents*observations.shape[1])].reshape(batch.batch_size, -1, self.mac.n_agents)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t) #$ HERE I SHOULD RUN ALL THE AGENTS CONCURRENTLY
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = mac_out.clone()[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = (1-self.alpha)*chosen_action_qvals + self.alpha * (rewards + self.args.gamma * (1 - terminated) * target_max_qvals)

        self.agent.update_qvalues(observations, actions, targets)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env