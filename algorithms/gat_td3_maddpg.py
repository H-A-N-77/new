import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Simple implementation of a Graph Attention layer.

    This implementation is intentionally lightweight and does not depend on
    external graph libraries. It assumes a fully-connected graph when the
    adjacency matrix has ones everywhere except for the diagonal.
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def _single_forward(self, h, adj):
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)

        # Compute attention coefficients
        a_input = torch.cat([
            Wh.repeat(1, N).view(N * N, -1),
            Wh.repeat(N, 1)
        ], dim=1).view(N, N, -1)
        e = self.leaky_relu(self.a(a_input).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def forward(self, h, adj):
        if h.dim() == 2:
            return self._single_forward(h, adj)
        outputs = []
        for i in range(h.size(0)):
            outputs.append(self._single_forward(h[i], adj))
        return torch.stack(outputs, dim=0)


class ActorGAT(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim_list, max_action):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.max_action = max_action
        self.gat1 = GraphAttentionLayer(obs_dim, 64)
        self.gat2 = GraphAttentionLayer(64, 64)
        self.action_heads = nn.ModuleList([
            nn.Linear(64, action_dim_list[i]) for i in range(num_agents)
        ])

    def forward(self, obs, adj):
        # obs: (batch, N, obs_dim)
        x = F.elu(self.gat1(obs, adj))
        x = F.elu(self.gat2(x, adj))
        actions = []
        for i in range(self.num_agents):
            a = torch.tanh(self.action_heads[i](x[:, i, :])) * self.max_action
            actions.append(a)
        return torch.stack(actions, dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x), inplace=True)
        x = F.relu(self.l2(x), inplace=True)
        return self.l3(x)


class MA_GAT_TD3(object):
    """Multi-agent TD3 with a shared GAT-based actor."""

    def __init__(self, num_agents, obs_dim_list, state_dim, action_dim_list,
                 max_action, device, min_adv_c=0.0, max_adv_c=1.0,
                 min_mi_c=0.0, max_mi_c=1.0, n_step=3, gamma=0.95,
                 tau=0.005, policy_freq=2):
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim_list[0]
        self.action_dim_list = action_dim_list
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.n_step = n_step

        # shared actor and target
        self.actor = ActorGAT(num_agents, self.obs_dim, action_dim_list, max_action).to(device)
        self.actor_target = ActorGAT(num_agents, self.obs_dim, action_dim_list, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # two critics and targets
        self.critic1 = Critic(state_dim, sum(action_dim_list)).to(device)
        self.critic2 = Critic(state_dim, sum(action_dim_list)).to(device)
        self.critic_target1 = Critic(state_dim, sum(action_dim_list)).to(device)
        self.critic_target2 = Critic(state_dim, sum(action_dim_list)).to(device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) +
                                                list(self.critic2.parameters()), lr=1e-3)

        # adjacency matrix for fully-connected graph
        adj = torch.ones(num_agents, num_agents) - torch.eye(num_agents)
        self.adj = adj.to(device)

        # dynamic coefficients
        self.min_adv_c = min_adv_c
        self.max_adv_c = max_adv_c
        self.min_mi_c = min_mi_c
        self.max_mi_c = max_mi_c
        self.adv_c = min_adv_c
        self.mi_c = min_mi_c
        self.total_it = 0

    def select_joint_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t, self.adj)[0].cpu().numpy()
        return action

    def update_dynamic_params(self):
        # Exponential approach to max values
        self.adv_c = self.max_adv_c - (self.max_adv_c - self.min_adv_c) * math.exp(-self.total_it / 1e5)
        self.mi_c = self.max_mi_c - (self.max_mi_c - self.min_mi_c) * math.exp(-self.total_it / 1e5)

    def train(self, replay_buffer, iterations=100, batch_size=100):
        for it in range(iterations):
            self.total_it += 1
            o, x, y, o_, u, r, d = replay_buffer.sample_n_step(
                batch_size, self.n_step, self.gamma)

            state = torch.FloatTensor(o.reshape(batch_size, -1)).to(self.device)
            next_state = torch.FloatTensor(o_.reshape(batch_size, -1)).to(self.device)
            action = torch.FloatTensor(u.reshape(batch_size, -1)).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)

            with torch.no_grad():
                next_obs = torch.FloatTensor(o_).to(self.device)
                next_action = self.actor_target(next_obs, self.adj).view(batch_size, -1)
                target_Q1 = self.critic_target1(torch.cat([next_state, next_action], 1))
                target_Q2 = self.critic_target2(torch.cat([next_state, next_action], 1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * (self.gamma ** self.n_step)) * target_Q

            current_Q1 = self.critic1(torch.cat([state, action], 1))
            current_Q2 = self.critic2(torch.cat([state, action], 1))
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.policy_freq == 0:
                obs_tensor = torch.FloatTensor(o).to(self.device)
                actor_action = self.actor(obs_tensor, self.adj).view(batch_size, -1)
                actor_loss = -self.critic1(torch.cat([state, actor_action], 1)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.update_dynamic_params()

    # Compatibility helpers for evaluation scripts
    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass
