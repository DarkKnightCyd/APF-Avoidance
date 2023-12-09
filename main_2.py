import numpy as np
from TD3Model import AgentTD3, ReplayBuffer
import matplotlib.pyplot as plt
import pandas as pd
from origin_apf_test import distanceCost, Traditional_Apf_Method
from Method import setup_seed
import random
import torch
import open3d as o3d
import os
from tqdm import tqdm
import copy
import time
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
obs_dim0 = np.array([200, 3])
goal_dim = 100
attraction_eta_dim = 50
step_size_eta_dim = 50
act_dim = obs_dim0[0] + attraction_eta_dim + step_size_eta_dim
total_input_dim = (obs_dim0[0] + goal_dim) * obs_dim0[1]
batch_size = 4096
if __name__ == '__main__':

    # ------------------------------PARAMETERS---------------------------------------

    setup_seed(3)  # 设置随机数种子
    env_path = r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-22-13/pointcloud-unity.ply"
    from_zero_start = True
    traj_path = r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-22-13/reference_trajectory.csv"
    reference_traj = pd.read_csv(
        traj_path,
        sep=',', header=None)
    reference_traj = reference_traj[1:].to_numpy(dtype=float)
    positon = reference_traj[1:, 1:4]
    tmp_end = np.array([-8.5, 21., 2.3])
    apf = Traditional_Apf_Method(env_path=env_path, start=positon[0], end=positon[-1],
                                 step_size_dim=step_size_eta_dim, attraction_dim=attraction_eta_dim)
    Agent = AgentTD3(using_dense=False, batch_size=batch_size)
    Agent.init(512, total_input_dim, act_dim, step_size_eta_dim, attraction_eta_dim)
    root_path = os.path.abspath(os.path.join(env_path, ".."))
    # ------------------------------PARAMETERS---------------------------------------

    # -------------------------------optional settings---------------------------------

    if not from_zero_start:
        root_path = os.path.abspath(os.path.join(env_path, ".."))
        model_path = "TrainedModel/agent_975"
        Agent.act.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/act.pkl", map_location=torch.device("cuda")))
        # Agent.act = torch.nn.DataParallel(Agent.act, device_ids=[0, 1], output_device=1)
        Agent.act_target.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/act_target.pkl", map_location=torch.device("cuda")))
        Agent.cri.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/cri.pkl", map_location=torch.device("cuda")))
        Agent.cri_target.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/cri_target.pkl", map_location=torch.device("cuda")))
        print('\n', "------ existing model loaded !!!-----------", '\n')

    # -------------------------------optional settings---------------------------------

    buffer = ReplayBuffer(int(1e6), total_input_dim, act_dim, False, True)
    gamma = 0.999
    MAX_EPISODE = 30000
    MAX_STEP = 100
    rand_act_episode_num = np.inf
    ep2_num = -np.inf
    update_cnt = 0
    rewardList = []
    maxReward = -np.inf
    save_cnt = 0
    save_cnt2 = 0
    buffer_full_flag = False
    for episode in range(MAX_EPISODE):
        # ------------------------RANDOM START AND END--------------------------
        while True:
            start_pos = [random.uniform(positon[0][0] - 5, positon[0][0] + 25),
                         random.uniform(positon[0][1] - 5, positon[0][1] + 5), positon[0][2]]
            end_pos = [random.uniform(positon[-1][0] - 25, positon[-1][0]),
                       random.uniform(positon[-1][1] - 5, positon[-1][1] + 5), positon[0][2]]
            start_pos = np.array(start_pos)
            end_pos = np.array(end_pos)
            f1, _ = apf.check_collison(start_pos)
            f2, _ = apf.check_collison(end_pos)
            if not f1 and not f2 \
                    and np.linalg.norm(start_pos - end_pos) > 30:
                # print("start_pos is {},".format(start_pos), '\n', "end_pos is {}".format(end_pos))
                apf.start = start_pos
                apf.end = end_pos
                break
        # ------------------------RANDOM START AND END--------------------------
        pos_now = apf.start
        rewardSum = 0
        pos_prev = [None, None, None]
        path_list = [apf.start]
        if episode >= rand_act_episode_num:
            Agent.explore_noise *= 0.999
        for j in range(MAX_STEP):
            # ----------get observation--------------
            obs_list = []
            r = apf.pcd_tree.search_radius_vector_3d(pos_now, apf.r0)
            indexes = r[1]
            for i in indexes:
                if apf.pcd.points[i][2] >= 3.1 or apf.pcd.points[i][2] <= 0.2: continue
                obs_list.append(apf.total_point_list[i])

            obs_list = np.array(obs_list)

            vec_end = np.array(apf.end - pos_now)

            if 1 <= obs_list.shape[0] <= obs_dim0[0]:
                state_list = copy.deepcopy(obs_list)
                state_list = np.vstack((state_list, np.zeros([obs_dim0[0] - obs_list.shape[0], 3])))
            elif obs_list.shape[0] < 1:
                state_list = np.zeros([obs_dim0[0], 3])
            elif obs_list.shape[0] > obs_dim0[0]:
                state_list = copy.deepcopy(obs_list[0:obs_dim0[0], :])
                obs_list = obs_list[0:obs_dim0[0], :]
            state_list = np.vstack((state_list, np.array([apf.end] * goal_dim).reshape(-1, 3)))

            state_list = state_list.reshape([1, -1])
            state_list = np.squeeze(state_list)
            if from_zero_start:
                if episode > ep2_num:
                    action = Agent.select_action(state_list)
                    # assert action.shape[0]==act_dim
                else:
                    action = np.array(
                        [random.uniform(Agent.rep_min, Agent.rep_max) for i in range(Agent.repusion_dim)] +
                        [random.uniform(Agent.step_min, Agent.step_max) for i in range(Agent.step_size_dim)] +
                        [random.uniform(Agent.attraction_min, Agent.attraction_max) for i in
                         range(Agent.attraction_dim)]
                        )

            else:
                action = Agent.select_action(state_list)

            pos_next, step_size_now = apf.get_next_point(pos_now, pos_prev, obs_list, action, True)
            path_list.append(pos_next)
            try:
                flag, D_MIN = apf.check_collison(pos_next)
            except TypeError:
                print("error!!!", flag, D_MIN)
                exit(0)
            reward, flag_done, dist1, dist2 = apf.getReward(flag, pos_prev, pos_now, pos_next, step_size_now, D_MIN)

            if reward == -3000:
                reward = reward + 3000 - 30
            elif reward == 3000:
                reward = reward - 3000 + 30
            rewardSum += reward
            done = True if ((dist1 <= apf.threshold) or flag_done or flag) else False
            mask = 0.0 if flag_done else gamma
            other = (reward, mask, *action)
            buffer.append_buffer(state_list, other)
            # if episode >= 50 and j % update_every == 0:
            #     Agent.update_net(buffer, update_every, batch_size, 1)
            #     update_cnt += update_every

            if buffer.next_idx_1 + buffer.next_idx_0 > 5000:
                if not buffer_full_flag:
                    print('\n', "*-*-*-*-buffer-5000-*-*-*-*-*-*", '\n')
                    rand_act_episode_num = episode
                    ep2_num = episode
                    buffer_full_flag = True

                Agent.update_net(buffer, batch_size)

            if done: break

            pos_prev = pos_now
            pos_now = pos_next
        end_dist = distanceCost(apf.end, pos_now)
        print('Episode:', episode, 'Reward:%f' % rewardSum, 'steps_total:', j, ' dist:%f' % end_dist)
        rewardList.append(round(rewardSum, 2))

        if episode > 500:

            if end_dist < 1 and save_cnt < 20:
                if save_cnt >= 20:
                    dir_list = os.listdir(root_path)
                    dir_list = sorted(dir_list, key=lambda file: os.path.getctime(os.path.join(save_path, file)),
                                      reverse=True)
                    dir_list = [f for f in dir_list if f.endswith(".csv") and f.startswith("path")]
                    dir_list = dir_list[0:save_cnt2]

                    os.remove(os.path.join(root_path, dir_list[-1]))
                np.savetxt(os.path.abspath(os.path.join(env_path, "..")) + r"/path_{}.csv".format(episode), path_list,
                           delimiter=',')
                save_cnt += 1
            if rewardSum > maxReward and (end_dist < 1):
                if save_cnt2 >= 10:
                    save_path = os.path.join(root_path, "TrainedModel")
                    dir_list = os.listdir(save_path)
                    dir_list = sorted(dir_list, key=lambda file: os.path.getctime(os.path.join(save_path, file)),
                                      reverse=True)
                    dir_list = dir_list[0:10]
                    shutil.rmtree(os.path.join(save_path, dir_list[-1]))

                maxReward = rewardSum
                os.mkdir(os.path.abspath(os.path.join(env_path, "..")) + r"/TrainedModel/agent_{}".format(episode))
                torch.save(Agent.act.state_dict(),
                           os.path.abspath(os.path.join(env_path, "..")) + r"/TrainedModel/agent_{}/act.pkl".format(
                               episode))
                torch.save(Agent.act_target.state_dict(),
                           os.path.abspath(
                               os.path.join(env_path, "..")) + r"/TrainedModel/agent_{}/act_target.pkl".format(
                               episode))
                torch.save(Agent.cri.state_dict(),
                           os.path.abspath(os.path.join(env_path, "..")) + r"/TrainedModel/agent_{}/cri.pkl".format(
                               episode))
                torch.save(Agent.cri_target.state_dict(),
                           os.path.abspath(
                               os.path.join(env_path, "..")) + r"/TrainedModel/agent_{}/cri_target.pkl".format(
                               episode))
                np.savetxt(os.path.abspath(os.path.join(env_path, "..")) + r"/TrainedModel/path_{}.csv".format(
                    episode), path_list,
                           delimiter=',')

                print('\n', 'reward={} is optimal now! model saved!'.format(rewardSum), '\n')
                save_cnt2 += 1
        # -------------visualize------------------

        # if episode>200 and episode%1==0 :
        #     plt.clf()
        #     plt.plot([i for i in range(len(rewardList))],rewardList)
        #     plt.pause(0.01)

    np.savetxt(os.path.abspath(os.path.join(env_path, "..")) + r"/rewardList_{}".format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))), rewardList, delimiter=',')
