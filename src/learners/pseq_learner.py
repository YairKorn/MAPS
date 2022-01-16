import numpy as np
from numpy.core.fromnumeric import cumsum
import torch as th
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch

class PSeqLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        print(f'### Learner uses TD-n ###')
        self.buffer = self.mac.action_model.buffer
        self.train_device = "cuda" if args.use_cuda else "cpu"
        self.device = self.buffer.device
        self.n_agents = args.n_agents
        self.td_n = self.n_agents # TD-n using n=number of agents

    """ A learner of PSeq architecture """
    # def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
    #     # if buffer in CPU and train uses CUDA, move the buffer to CUDA
    #     if self.episode_buffer.device != self.train_device:
    #         self.episode_buffer.to(self.train_device)

    #     # use the basic q-learner but episodes are taken from the internal, decomposed, buffer
    #     super().train(self.episode_buffer, t_env, episode_num)

    #     # if buffer in should be in CPU, return the buffer from CUDA to the CPU
    #     if self.episode_buffer.device != self.buffer_device:
    #         self.episode_buffer.to(self.buffer_device)

    # def gamma(self, l=1):
    #     g = th.cat((th.ones((1, self.args.n_agents-1)), self.args.gamma*th.ones((1, 1))), axis=1)
    #     return g.repeat(1, int(l/self.args.n_agents)+1)[0, :l]


    def train(self, _: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = self.buffer["reward"][:, :-1]
        actions = self.buffer["actions"][:, :-1]
        terminated = self.buffer["terminated"][:, :-1].float()
        mask = self.buffer["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = self.buffer["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(self.buffer.batch_size)
        for t in range(self.buffer.max_seq_length):
            agent_outs = self.mac.forward(self.buffer, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate TD-n, for n agents
        #! TODO: allegedly, reward_conv's gamma element depends on the number of the agent
        reward_conv = th.nn.Conv1d(1, 1, self.td_n, bias=False, device=rewards.device)
        reward_conv.weight.data = th.ones((1, 1, self.td_n)) # th.cat((th.ones((1, 1, self.td_n-1)), self.gamma * th.ones(1, 1, 1)), axis=2)
        rewards = reward_conv(th.cat((rewards.reshape(self.buffer.batch_size, 1, -1), th.zeros(self.buffer.batch_size, 1, self.td_n-1)), axis=2))
        rewards = rewards.reshape(self.buffer.batch_size, -1, 1)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(self.buffer.batch_size)
        for t in range(self.buffer.max_seq_length):
            target_agent_outs = self.target_mac.forward(self.buffer, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[self.td_n:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, self.td_n:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, self.td_n:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = th.cat((target_max_qvals, th.zeros((32, self.td_n-1, 1))), axis=1)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, self.buffer["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, self.buffer["state"][:, 1:])

        # Mask "terminated"
        mask_ind = th.where(terminated)[1]
        for ind in range(self.buffer.batch_size):
            terminated[ind, (mask_ind[ind]-self.td_n+1):mask_ind[ind], 0] = 1


        # Calculate n-step Q-Learning targets
        #! gamma should be reduced in the last episodes
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
