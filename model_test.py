import torch
import numpy as np
from copy import deepcopy
import open3d as o3d
import random
import os
import time
import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from origin_apf_test import Traditional_Apf_Method,distanceCost
from TD3Model import AgentTD3,Actor,Actor2
from collections import OrderedDict
#from main_2 import batch_size,obs_dim0,obs_dim1,act_dim
obs_dim0=np.array([200,3])
goal_dim = 100
attraction_eta_dim=50
step_size_eta_dim=50
act_dim=obs_dim0[0]+attraction_eta_dim+step_size_eta_dim
total_input_dim= (obs_dim0[0]+goal_dim) * obs_dim0[1]
batch_size=4096


def pre_process(original_dict):
    #from module to no module
    new_state_dict = OrderedDict()
    for key, value in original_dict.items():
        name = key[0:4] + key[11:]
        new_state_dict[name] = value
    return new_state_dict

def check_valid(pcd,pos):
    r = o3d.geometry.KDTreeFlann(pcd).search_knn_vector_3d(pos,1)
    dmin = np.linalg.norm(pcd.points[r[1][0]] - pos)
    return False if dmin<=0.3 else True

def visualize_and_collison_check(path_to_show,env_pointcloud):

    if path_to_show is not None:
        path_to_show = np.array(path_to_show)
        #print(path_to_show.shape[0])
        plt.subplot(211)
        plt.plot(path_to_show[1:,0],path_to_show[1:,1],c='red',marker='o',linewidth=0.3)

        plt.subplot(212)
        plt.plot([i for i in range(path_to_show.shape[0])] ,path_to_show[:,2])
        plt.show()
        env2 = deepcopy(env_pointcloud)
        env_tree = o3d.geometry.KDTreeFlann(env2)
        pos_converted = o3d.utility.Vector3dVector(path_to_show)
        new_env = o3d.open3d.geometry.PointCloud()
        points_for_newenv = []
        colors_path = np.array([np.array([1., 0., 1.]) for _ in pos_converted])
        for idx, point in enumerate(env2.points):
            if point[2] > 3.:
                continue
            else:
                points_for_newenv.append(point)
    tmp = np.array([[-8.5, 21.]*80]).reshape(-1,2)
    z = np.expand_dims(np.array([-0.3+i*0.1 for i in range(80)]),1)
    tmp = np.hstack((tmp,z))
    new_env = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_for_newenv))


    only_path = o3d.geometry.PointCloud()
    only_path.points = o3d.utility.Vector3dVector(np.array(path_to_show))
    only_path.paint_uniform_color([0,0,0])
    o3d.visualization.draw_geometries([new_env,only_path])
    o3d.visualization.draw_geometries([only_path])

    t = []
    for idx, p in enumerate(path_to_show):
        res = env_tree.search_radius_vector_3d(p, radius=0.3)
        d_list = res[-1]
        if not d_list:
            continue
        else:
            print("invalid!!!", np.min(d_list), idx, p)
            t.append(res)

if __name__ == "__main__":

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attraction_eta_dim = 50
    step_size_eta_dim = 50
    root_path= r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-22-13/"
    #env_path = os.path.join(root_path,"pointcloud-unity.ply")
    env_path  =os.path.join(r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-22-13/","pointcloud-unity.ply")
    traj_path = os.path.join(root_path,"reference_trajectory.csv")
    model_root_path=r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-22-13/"
    model_path="TrainedModel/agent_635"
    path_name = "TrainedModel/path_3911.csv"
    x = pre_process(torch.load(os.path.join(model_root_path, model_path)
            + r"/act.pkl", map_location=torch.device("cuda")))
    try:

        path_exist=pd.read_csv(os.path.join(root_path,path_name),sep=',',header=None)

        if path_name.startswith("reference") or path_name.startswith("odometry"):
            path_exist=path_exist[1:]
            path_exist = path_exist.to_numpy(dtype=float)
            path_exist = path_exist[1:, 1:4]
        else:
            path_exist = path_exist.to_numpy(dtype=float)

    except FileNotFoundError:
        path_exist=[]


    reference_traj = pd.read_csv(
        traj_path,
        sep=',', header=None)
    reference_traj = reference_traj[1:].to_numpy(dtype=float)
    positon = reference_traj[1:, 1:4]
    env_pcd = o3d.io.read_point_cloud(env_path)

    while True:
        start_pos = [random.uniform(positon[0][0]-5,positon[0][0]+25),random.uniform(positon[0][1]-5,positon[0][1]+5),positon[0][2]]
        end_pos = [random.uniform(positon[-1][0]-25,positon[-1][0]),random.uniform(positon[-1][1]-5,positon[-1][1]+5),positon[0][2]]

        if check_valid(env_pcd,start_pos) and check_valid(env_pcd,end_pos) \
                and np.linalg.norm(np.array(start_pos)-np.array(end_pos))>30:
            print("start_pos is {},".format(start_pos),'\n',"end_pos is {}".format(end_pos))
            break
    apf = Traditional_Apf_Method(env_path=env_path, start=start_pos, end=positon[-1], step_size_dim=step_size_eta_dim,
                                 attraction_dim=attraction_eta_dim)
    # apf = Traditional_Apf_Method(env_path=r"/home/hit312/cyd/tmp/simple1.ply", start=np.array([-10, 22, 2]),
    #                              end=np.array([25, 22, 2]), step_size_dim=step_size_eta_dim,
    #                              attraction_dim=attraction_eta_dim)
    select_flag = 0


    if select_flag < 1:
        if not path_exist==[]:
            visualize_and_collison_check(path_exist, apf.pcd)
        #apf.cal_rewards_for_path(path_exist)
        else:
            print("path none!")

    else:

        Agent = Actor2(512, total_input_dim, act_dim, 50, 50,False)
        Agent.load_state_dict(
            pre_process(torch.load(os.path.join(model_root_path, model_path)
            + r"/act.pkl", map_location=torch.device("cuda"))))

        MAX_STEP=2000
        pos_now = apf.start
        pos_prev = [None, None, None]
        rewardSum=0
        step_size_list=[]
        Agent.step_max = 0.2

        start = time.perf_counter()
        for j in range(MAX_STEP):

            obs_list = []

            r = apf.pcd_tree.search_radius_vector_3d(pos_now, 0.8)
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
            action = Agent.select_action(state_list,False)
            pos_next,step_size = apf.get_next_point(pos_now, pos_prev, obs_list, np.squeeze(action), True)
            apf.path.append(pos_next)
            step_size_list.append(step_size)
            flag,D_MIN= apf.check_collison(pos_next)
            reward,flag_done,d1,d2 = apf.getReward(flag, pos_prev, pos_now, pos_next,step_size,D_MIN)
            rewardSum += reward
            #done = True if ((distanceCost(apf.end, pos_now) <= apf.threshold) or flag) else False
            done = True if ((distanceCost(apf.end, pos_now) <= apf.threshold) or flag_done or flag) else False
            #done = True if distanceCost(apf.end, pos_now) <= apf.threshold else False
            if done:

                break
            pos_prev = pos_now
            pos_now = pos_next

        # plt.plot(np.arange(len(t_list)),t_list)
        # plt.show()
        # print(t_list[-1])
        # exit(0)
        dt = time.perf_counter()-start
        print("all the verbose takes {} s".format(dt),'\n',
              "average 50 loops takes {} ms".format(dt/50*1000)
              )
        print(rewardSum)
        print(max(step_size_list),min(step_size_list),np.mean(step_size_list))
        visualize_and_collison_check(apf.path, apf.pcd)


