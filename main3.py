import numpy as np

from Model2 import *
from TD3Model import ReplayBuffer2, ReplayBuffer3
import matplotlib.pyplot as plt
import pandas as pd
from origin_apf_test import distanceCost, Traditional_Apf_Method, apf_env
from Method import setup_seed
import random
import torch
import open3d as o3d
import os
from tqdm import tqdm
import copy
import time
import shutil


def show_(pcd_origin, start, end):
    point = np.asarray([pt for pt in pcd_origin.points if pt[2] < 2.55])
    start = np.array([[*start[0:2], -2 + 0.1 * i + start[-1]] for i in range(21)])
    end = np.array([[*end[0:2], -2 + 0.1 * i + end[-1]] for i in range(21)])
    l3 = np.array([[0.9999992, 19.99999988, -2 + 0.1 * i + 1.8875645] for i in range(21)])
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((start, end, l3))))
    p.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (21 * 3, 1)))

    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(point)
    ), p])


if __name__ == '__main__':
    setup_seed(4)
    buffer = ReplayBuffer3(int(11e4), act_dim=150)

    from_zero_start = True
    agent = Agent3()

    pcd_path = r"./pcd/t3.ply"

    # start = np.array([-22.0000008, 13.99999988, 1.8875645])
    start = np.array([-4, 14.5, 1.5])
    # end = np.array([-8.9999992, 19.99999988, 1.8875645])
    end = np.array([-11, 7., 1.5])
    env = apf_env(
        pcd_path, start, end
    )
    # show_(env.pcd, start, end)

    if not from_zero_start:
        root_path = r"./models"
        model_path = ""
        agent.actor.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/act.pkl", map_location=torch.device("cuda")))
        # Agent.act = torch.nn.DataParallel(Agent.act, device_ids=[0, 1], output_device=1)
        agent.target_actor.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/act_target.pkl", map_location=torch.device("cuda")))
        agent.critic1.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/critic1.pkl", map_location=torch.device("cuda")))
        agent.critic2.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/critic2.pkl", map_location=torch.device("cuda")))
        agent.target_critic1.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/target_critic1.pkl", map_location=torch.device("cuda")))
        agent.target_critic2.load_state_dict(
            torch.load(os.path.join(root_path, model_path) + r"/target_critic2.pkl", map_location=torch.device("cuda")))
        print('\n', "------ existing model loaded !!!-----------", '\n')
    else:
        MAX_EPISODE = 10000
        MAX_STEP = 200
        batch_size = 1024 * 4
        gamma = 0.98
        rand_act_episode_num = np.inf
        ep2_num = -np.inf
        update_cnt = 0
        rewardList = []
        maxReward = -np.inf
        save_cnt = 0
        save_cnt2 = 0
        buffer_full_flag = False
        time1 = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
        os.mkdir("./models/" + time1)
        save_path = os.path.abspath("./models/" + time1)

        for episode in range(MAX_EPISODE):
            done = False
            observation = env.reset()
            pos_now = env.start

            rewardSum = 0
            step_cnt = 0
            if episode > 499:
                agent.policy_noise *= 0.9999
            for step_cnt in range(MAX_STEP):

                # obs = env.get_observation(env.pos_now)
                if from_zero_start:
                    if episode > ep2_num:
                        action = agent.select_action(env.observation_now, True)
                        # assert action.shape[0]==act_dim
                    else:
                        action = np.array([random.uniform(-1, 1) for i in range(50)])

                else:
                    action = agent.select_action(env.observation_now, True)
                obs0 = env.observation_now
                observation_, reward, done1 = env.step(action)
                rewardSum += reward
                done = True if (done1 or distanceCost(env.path[-1], env.end) < 1) else False
                mask = 0.0 if done else gamma

                buffer.append_buffer((obs0, action, reward, observation_, mask))

                if buffer.next_idx_1 > 2500 and buffer.next_idx_0 > 2500 and episode > 499:
                    if not buffer_full_flag:
                        print('\n', "*-*-*-*-buffer-5000-*-*-*-*-*-*", '\n')
                        rand_act_episode_num = episode
                        ep2_num = episode
                        buffer_full_flag = True
                    agent.update_net(buffer, batch_size)

                # print(pos_now, np.linalg.norm(pos_now - env.path[-2]), len(env.path))
                if done or step_cnt == MAX_STEP - 1:
                    def show2_():
                        point = np.asarray([pt for pt in env.pcd.points if pt[2] < 2.55])
                        start = np.array([[*env.start[0:2], -2 + 0.1 * i + env.start[-1]] for i in range(21)])
                        end = np.array([[*env.end[0:2], -2 + 0.1 * i + env.end[-1]] for i in range(21)])
                        path = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                            np.vstack((np.asarray(env.path), start, end))
                        ))
                        path.colors = o3d.utility.Vector3dVector(
                            np.asarray(np.tile([0, 0, 0], (len(path.points), 1)))
                        )
                        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point)),
                                                           path])


                    # show2_()
                    # if done:
                    #     print(env.path[-1],env.check_collision(env.path[-1]))
                    break
                step_cnt += 1
            end_dist = distanceCost(env.end, env.pos_now)
            print('Episode:', episode, 'Reward:%f' % rewardSum, ' dist:%f' % end_dist, "step:{}".format(step_cnt))
            rewardList.append(round(rewardSum, 2))
            # -------------------SAVINGS------------------

            if episode > 499:

                if rewardSum > maxReward and (end_dist < 1):
                    if save_cnt2 >= 3:
                        dir_list = os.listdir(save_path)
                        dir_list = [p for p in dir_list if p.startswith("opt")]
                        dir_list = sorted(dir_list, key=lambda file: os.path.getctime(os.path.join(save_path, file)),
                                          reverse=True)
                        dir_list = dir_list[0:10]
                        shutil.rmtree(os.path.join(save_path, dir_list[-1]))
                    maxReward = rewardSum
                    os.mkdir(save_path + r"/opt_{}".format(episode))
                    current_folder = save_path + r"/opt_{}".format(episode)
                    torch.save(agent.actor.state_dict(),
                               current_folder + r"/actor.pkl")
                    torch.save(agent.target_actor.state_dict(),
                               current_folder + r"/target_actor.pkl")
                    torch.save(agent.target_critic1.state_dict(),
                               current_folder + r"/target_critic1.pkl")
                    torch.save(agent.target_critic2.state_dict(),
                               current_folder + r"/target_critic2.pkl")
                    np.savetxt(current_folder + r"/path.csv", env.path,
                               delimiter=',')

                    print('\n', 'reward={} is optimal now! model saved!'.format(rewardSum), '\n')
                    save_cnt2 += 1

                elif episode % 1000 == 0:
                    os.mkdir(save_path + r"/loop_{}".format(episode))
                    current_folder = save_path + r"/loop_{}".format(episode)
                    torch.save(agent.actor.state_dict(),
                               current_folder + r"/actor.pkl".format(
                                   episode))
                    torch.save(agent.target_actor.state_dict(),
                               current_folder + r"/target_actor.pkl".format(
                                   episode))
                    torch.save(agent.target_critic1.state_dict(),
                               current_folder + r"/target_critic1.pkl".format(
                                   episode))
                    torch.save(agent.target_critic2.state_dict(),
                               current_folder + r"/target_critic2.pkl".format(
                                   episode))
                    np.savetxt(current_folder + r"/path.csv", env.path,
                               delimiter=',')

                if episode % 500 == 0 or episode == MAX_EPISODE - 1:
                    np.savetxt(save_path + r"/reward_list_0-{}".format(episode), rewardList, delimiter=',')
    print("f")
