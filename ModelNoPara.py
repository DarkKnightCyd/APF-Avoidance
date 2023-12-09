# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import torch.nn as nn

import TD3Model

class ReplayBuffer_No_Para:
    def __init__(self, max_size):
        self.device = torch.device("cuda")
        self.max_len = max_size
        self.now_len_0 = 0
        self.now_len_1 = 0
        self.next_idx_0 = 0
        self.next_idx_1 = 0
        self.if_full_0 = False
        self.if_full_1 = False

        self.state_a_0 = torch.empty((max_size, 500, 3), dtype=torch.float32, device='cuda:0')
        self.state_b_0 = torch.empty((max_size, 6), dtype=torch.float32, device='cuda:0')

        self.state_next_a_0 = torch.empty((max_size, 500, 3), dtype=torch.float32, device='cuda:0')

        self.state_next_b_0 = torch.empty((max_size, 6), dtype=torch.float32, device='cuda:0')

        self.action_0 = torch.empty((max_size, 52), dtype=torch.float32, device='cuda:0')

        self.reward_0 = torch.empty((max_size, 1), dtype=torch.float32, device='cuda:0')

        self.done_0 = torch.empty((max_size, 1), dtype=torch.bool, device='cuda:0')


    def append_buffer(self, SARSD: tuple):
        if self.if_full_0 :
            return
        state, action, reward, state_next, done = SARSD
        state_a, state_b = state
        state_next_a, state_next_b = state_next

        dev = torch.device('cuda:0')
        state_a = torch.as_tensor(state_a).to(dev)
        state_b = torch.as_tensor(state_b).to(dev)
        state_next_a = torch.as_tensor(state_next_a).to(dev)
        state_next_b = torch.as_tensor(state_next_b).to(dev)
        action = torch.as_tensor(action).to(dev)
        reward = torch.as_tensor(reward).to(dev)
        done = torch.as_tensor(done).to(dev)
        self.state_a_0[self.next_idx_0] = state_a
        self.state_b_0[self.next_idx_0] = state_b
        self.state_next_a_0[self.next_idx_0] = state_next_a
        self.state_next_b_0[self.next_idx_0] = state_next_b
        self.action_0[self.next_idx_0] = action
        self.reward_0[self.next_idx_0] = reward
        self.done_0[self.next_idx_0] = done
        self.next_idx_0 += 1


        if self.next_idx_0 >= self.max_len:
            self.if_full_0 = True
            self.next_idx_0 = 0


    def sample_batch(self, batch_size):
        # print(self.now_len_0, self.now_len_1)

        indices = torch.as_tensor(np.random.choice(self.now_len_0 - 1, (batch_size,), replace=False),
                                  ).to(torch.device('cuda:0'))
        s_a, s_b = self.state_a_0[indices], self.state_b_0[indices]
        # s_a:(bs,1000,3) s_b:(bs,6)
        s = (s_a, s_b)

        a = self.action_0[indices]
        r = self.reward_0[indices]
        s_next_a, s_next_b = self.state_next_a_0[indices], self.state_next_b_0[indices]
        s_ = (s_next_a, s_next_b)

        d = self.done_0[indices]
        return (s, a, r, s_, d)


    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len_0 = self.max_len if self.if_full_0 else self.next_idx_0
        self.now_len_1 = self.max_len if self.if_full_1 else self.next_idx_1


class T_Net(nn.Module):
    def __init__(self):
        super(T_Net, self).__init__()
        # 这里需要注意的是上文提到的MLP均由卷积结构完成
        # 比如说将3维映射到64维，其利用64个1x3的卷积核
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # x = self.relu(self.bn4(self.fc1(x)))
        # x = self.relu(self.bn5(self.fc2(x)))
        # x = self.fc3(x)
        #
        # x = x.view(-1, 3, 3)  # 输出为Batch*3*3的张量
        return x


class Actor_MultiModal(nn.Module):
    def __init__(self):
        super(Actor_MultiModal, self).__init__()
        self.branch_net1 = T_Net().cuda()
        self.branch_net2 = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 200), nn.ReLU(),
            nn.LayerNorm(200), nn.ReLU(),
        ).cuda()

        self.net3 = nn.Sequential(
            nn.Linear(1224, 512), nn.ReLU(),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 52), nn.LayerNorm(52)
        ).cuda()


    def forward(self, state: tuple):
        x1, x2 = state
        # x1 (bs,1000,3) x2 (bs,6)

        x1 = x1.float().transpose(1, 2)
        x2 = x2.float()
        x1 = self.branch_net1(x1)  # (bs,x1)
        x2 = self.branch_net2(x2)  # (bs,x)
        return self.net3(torch.concat((x1, x2), 1).cuda()).tanh()

    def get_action(self, states: tuple, action_std, rand=True) -> torch.Tensor:

        states = (torch.unsqueeze(states[0], 0), torch.unsqueeze(states[1], 0))
        action = self.forward(states)[0]
        if rand:

            noise = torch.as_tensor(np.random.normal(0, action_std, action.shape), dtype=torch.float32).to(
                action.device)
            return (action + noise).clamp(-1.0, 1.0)
        else:

            return action


class Critic_MultiModal(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_net1 = T_Net().cuda()
        self.branch_net2 = nn.Sequential(
            nn.Linear(6 + 52, 128), nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 200), nn.LayerNorm(200),
            nn.ReLU(),
        ).cuda()

        self.net3 = nn.Sequential(
            nn.Linear(1224, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 1),
        ).cuda()


    def forward(self, s_a):
        x1, x2 = s_a
        x1 = self.branch_net1(x1.float().transpose(1, 2))  # (bs,x1)
        x2 = self.branch_net2(x2.float())  # (bs,x)
        return self.net3(torch.concat((x1, x2), 1).cuda())


class Agent:
    def __init__(self,
                 gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2,
                 learning_rate=5e-5):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.actor = Actor_MultiModal()
        self.critic1 = Critic_MultiModal()
        self.critic2 = Critic_MultiModal()
        self.target_actor = Actor_MultiModal()
        self.target_critic1 = Critic_MultiModal()
        self.target_critic2 = Critic_MultiModal()
        self.cnt = 0
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.act_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.cri_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.cri_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

    def select_action(self, observation, train=False):
        self.actor.eval()
        # state = torch.as_tensor(observation, torch.device("cuda"))
        state = (torch.as_tensor(observation[0]).cuda(),
                 torch.as_tensor(observation[1]).cuda())
        if train:
            action = self.actor.get_action(
                state, 0.9, True
            )
        else:
            action = self.actor.get_action(state, 0, False)
        self.actor.train()

        return action.detach().cpu().numpy()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def update_net(self, buffer, batch_size):
        buffer.update_now_len_before_sample()
        self.cnt += 1
        obj_critic = obj_actor = None
        # for i in range(int(target_step * repeat_times)):
        '''objective of critic (loss function of critic)'''
        with torch.no_grad():
            state, action, reward, next_s, mask = buffer.sample_batch(batch_size)
            device = action.device
            tmp = self.target_actor.forward(next_s).to(device)

            noise = torch.normal(0., self.policy_noise, size=tmp.shape, device=device).clamp(-0.5, 0.5)

            next_a = (tmp + noise).clamp(-1, 1)
            q1_ = self.target_critic1.forward([
                next_s[0],
                torch.cat((
                    next_s[1], torch.squeeze(next_a, dim=1)
                ), dim=1)
            ])
            q2_ = self.target_critic2.forward([
                next_s[0],
                torch.cat((
                    next_s[1], torch.squeeze(next_a, dim=1)
                ), dim=1)
            ])
            idx = [i for i, done in enumerate(mask) if done]
            q1_[idx] = 0.
            q2_[idx] = 0.
            next_q = torch.min(q1_, q2_).to(device)
            # twin critics

            q_label = reward + mask * next_q

        q1 = self.critic1.forward(
            [
                state[0], torch.cat((state[1], action), dim=1)
            ]
        ).to(device)
        q2 = self.critic2.forward(
            [
                state[0], torch.cat((state[1], action), dim=1)
            ]
        ).to(device)

        critic_loss = self.criterion(q1, q_label.detach()) + self.criterion(q2, q_label.detach())

        self.cri_optimizer1.zero_grad()
        self.cri_optimizer2.zero_grad()
        critic_loss.backward()
        self.cri_optimizer1.step()
        self.cri_optimizer2.step()

        # ----------------------delay-update-------------------
        if self.cnt % self.delay_time == 0:  # delay update
            '''objective of actor'''

            q_value_pg = self.actor.forward(state)  # policy gradient
            actor_loss = -self.critic1.forward(
                [state[0].cuda(), torch.cat((state[1].cuda(), torch.squeeze(q_value_pg, dim=1).cuda()), dim=1)]
            ).mean()
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
