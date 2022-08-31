import torch as th
from .q_learner import QLearner
from components.episode_buffer import EpisodeBatch

import time

""" TDn Learner implements offline TD-lambda bounded by n-steps due to performances issues """
class TDnLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.device = "cuda" if args.internal_buffer_cuda else "cpu" #! Specific to MAPS; consider generalizing

        self.alg = args.RL_Algorithm

        if self.alg == "TD":
            # TD-n properties
            self.n_bound = args.TDn_bound if args.TDn_bound is not None else 1 # TD-n default n=1 (Q-learning)
            self.n_values = range(1, self.n_bound+1)
            self.n_weight = th.cat(((1 - args.TDn_weight) * (args.TDn_weight ** th.arange(self.n_bound-1)), \
                th.tensor([args.TDn_weight ** (self.n_bound-1)]))).to(self.device).view(-1, 1, 1, 1)
        elif self.alg == "MC":
            self.n_values = [args.env_args['episode_limit'] + 1, ]
            self.n_weight = th.ones(size=(1, 1, 1, 1)).to(self.device)

    """ An offline n-bound TD-lambda learner """
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        #$ TEST!!
        test_dic = th.zeros((self.mac.args.batch_size, self.mac.action_model.episode_limit * self.mac.n_agents + 1, self.mac.args.rnn_hidden_dim))

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.mac.forward(batch, t=t)
            test_dic[:, t, :] = self.mac.hidden_states.detach()
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            # test_dic[:, t, :] = self.target_mac.hidden_states.detach()
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        targets = []
        ### Calculate TD-n, for 1...TDn_bound ##
        for n in self.n_values:
            reward_conv = th.nn.Conv1d(1, 1, n, bias=False)
            reward_conv.weight.data = (self.args.gamma ** th.arange(n).view(1, 1, -1)).to(self.device)
            rewards_n = reward_conv(th.cat((rewards.reshape(batch.batch_size, 1, -1), th.zeros((batch.batch_size, 1, n-1), device=self.device)), axis=2))
            rewards_n = rewards_n.reshape(batch.batch_size, -1, 1)

            target_max_qvals_n = th.cat((target_max_qvals[:, n-1:], th.zeros((batch.batch_size, n-1, 1), device=self.device)), axis=1)

            # Mask "terminated"
            term_n = terminated.clone()
            mask_ind = th.where(term_n)
            for i in range(mask_ind[0].numel()):
                ep, ind = mask_ind[0][i], mask_ind[1][i]
                term_n[ep, max(ind-n+1, 0):ind, 0] = 1

            # Calculate n-step Q-Learning targets
            targets.append(rewards_n + (self.args.gamma ** n) * (1 - term_n) * target_max_qvals_n)

        targets = th.stack(targets, dim=0)
        targets = (self.n_weight * targets).sum(dim=0)  # Concat across TD-ns, and multiply by the weights

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        #$ #$ #$ QVALUES TEST $# $# $#
        # calc_qvalues = mac_out[0, :-1, 0, 6].cpu()
        # targ_qvalues = th.arange(self.mac.n_agents, 0, -1) * (self.mac.action_model.reward_hunt + self.mac.action_model.reward_catch)

        # q_loss = ((calc_qvalues - targ_qvalues) ** 2).sum() / calc_qvalues.numel()
        # print(f'Calc: {calc_qvalues.detach()}')
        # print(f'MC  : {targets[0, :, 0].detach()}')

        # time.sleep(0.2)

        #$ #$ #$   TILL HERE   $# $# $#
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
            #$ #$ QVALUES TEST $# $#
            # self.logger.log_stat("q_loss", q_loss.item(), t_env)
            # self.logger.log_stat("f_value", calc_qvalues[0].item(), t_env)
            # self.logger.log_stat("l_value", calc_qvalues[-1].item(), t_env)

            self.log_stats_t = t_env