import math
import os

import sys

import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def visualize_and_collison_check_0(path_to_show, env_pointcloud):
    path_to_show = np.array(path_to_show)
    if path_to_show.shape[0] < 1: return
    print(path_to_show.shape[0])
    plt.subplot(211)
    plt.plot(path_to_show[1:, 0], path_to_show[1:, 1], c='red', marker='o', linewidth=0.3)
    plt.subplot(212)
    plt.plot([i for i in range(path_to_show.shape[0])], path_to_show[:, 2])
    plt.show()
    env2 = deepcopy(env_pointcloud)
    env_tree = o3d.geometry.KDTreeFlann(env2)
    pos_converted = o3d.utility.Vector3dVector(path_to_show)
    new_env = o3d.open3d.geometry.PointCloud()
    points_for_newenv = []
    colors_path = np.array([np.array([1., 0., 1.]) for _ in pos_converted])
    for idx, point in enumerate(env2.points):
        if point[2] > 3.5:
            continue
        else:
            points_for_newenv.append(point)

    # tmp = np.array([[-8.5, 21.]*80]).reshape(-1,2)
    # z = np.expand_dims(np.array([-0.3+i*0.1 for i in range(80)]),1)
    # tmp = np.hstack((tmp,z))

    new_env.points = o3d.utility.Vector3dVector(np.vstack((points_for_newenv, np.array(path_to_show))))
    o3d.visualization.draw_geometries([new_env])
    only_path = o3d.geometry.PointCloud()
    only_path.points = o3d.utility.Vector3dVector(np.array(path_to_show))
    o3d.visualization.draw_geometries([only_path])
    # ----------------collision check-----------------
    t = []
    for idx, p in enumerate(path_to_show):
        res = env_tree.search_radius_vector_3d(p, radius=0.3)
        d_list = res[-1]
        if not d_list:
            continue
        else:
            print("invalid!!!", np.linalg.norm(env2.points[res[1][0]] - p), idx, p)
            t.append(res)


def distanceCost(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def get_diff_vec(pos1, pos2):
    return (pos1 - pos2) / distanceCost(pos1, pos2)


def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
    temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
    temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
    theta = np.arccos(temp)
    return theta


def cal_angle(v1, v2):
    x = np.dot(v1, v2)
    y1 = np.linalg.norm(v1)
    y2 = np.linalg.norm(v2)

    ans = x / (y1 * y2)
    ans = np.clip(ans, -1, 1)
    return np.arccos(ans) / np.pi * 180


def get_norm_vec(v1):
    f1 = abs(math.atan2(v1[1], v1[0]) * 180 / math.pi) < 1e-3 or abs(
        180 - math.atan2(v1[1], v1[0]) * 180 / math.pi) < 1e-3
    f2 = abs(90 - abs(math.atan2(v1[1], v1[0]) * 180 / math.pi)) < 1e-3
    if not (f1 or f2):
        ret = np.array([1, -v1[0] / v1[1]])
        return ret / np.linalg.norm(ret)
    elif f1:
        return np.array([0, 1])
    elif f2:
        return np.array([1, 0])


class Traditional_Apf_Method:
    def __init__(self, env_path, start, end, step_size_dim, attraction_dim):
        print("________loading point cloud env from path: ", env_path)
        self.pcd = o3d.io.read_point_cloud(env_path)
        # self.pcd=self.pcd.voxel_down_sample(voxel_size=0.5)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.total_point_list = np.array(self.pcd.points)
        self.end = end  # 目标点
        self.start = start  # 轨迹起始点
        self.stepSize = 0.1  # 物体移动的固定步长
        self.dgoal = 10  # 当q与qgoal距离超过它时将衰减一部分引力
        self.r0 = 1  # 斥力超过这个范围后将不复存在
        self.threshold = 0.5  # q与qgoal距离小于它时终止训练或者仿真
        self.UAV_safe_radius = 0.3
        # -------------路径（每次getqNext会自动往path添加路径）---------#
        self.path = [self.start.copy()]
        # ------------运动学约束------------#
        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 30 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 30 / 180 * np.pi  # 最大俯冲角
        self.num_of_obs = []
        self.epsilon0 = 0.8
        self.attraction_dim = attraction_dim
        self.step_size_dim = step_size_dim
        self.y_low_bound = 10
        self.y_high_bound = 30
        self.x_low_bound = -30
        self.x_high_bound = 50
        self.rep_max = -np.inf
        self.attraction_max = -np.inf
        self.temp_traj = [self.start]

    def reset(self):
        self.path = [self.start.copy()]
        del self.temp_traj[1:]

    def attraction(self, pos_now, epsilon):  # 计算引力的函数
        dist = distanceCost(pos_now, self.end)
        vec = self.end - pos_now
        if dist <= self.dgoal:
            f = epsilon * vec
        else:
            f = epsilon * self.dgoal * vec / dist
        self.attraction_max = max(np.linalg.norm(f), self.attraction_max)
        return np.array(f)

    def attraction_v2(self, pow_now, epsilon):
        return get_diff_vec(self.end, pow_now) * epsilon

    def repulsion_one_only(self, pos_now, eta, obs_pos):
        dist = distanceCost(pos_now, obs_pos)
        diff_vec = get_diff_vec(pos_now, obs_pos)
        if dist <= self.r0:
            f = eta * (1 / dist - 1 / self.r0) * (dist / self.r0) ** 2 * diff_vec + eta * (
                    1 / dist - 1 / self.r0) ** 2 * dist * diff_vec

            f = np.array(f)
        else:
            f = np.zeros(3)
        self.rep_max = max(self.rep_max, np.linalg.norm(f))
        return f

    def repulsion_one_only_v2(self, pow_now, eta, obs_pos):
        return get_diff_vec(pow_now, obs_pos) * eta

    def get_total_force(self, pos_now, obs_list, eta_list):
        attraction_force = self.attraction(pos_now, epsilon=np.mean(eta_list[-self.attraction_dim:]))
        rep_force = np.zeros(3)
        for idx, obs in enumerate(obs_list):
            rep_force += self.repulsion_one_only(pos_now, eta_list[idx], obs)
        total_foce = rep_force + attraction_force

        return total_foce / np.linalg.norm(total_foce)

    def check_collison(self, pos):

        if not (self.x_low_bound <= pos[0] <= self.x_high_bound and self.y_low_bound <= pos[
            1] <= self.y_high_bound and 0.5 <= pos[2] <= 3):
            return True, None
        else:
            # r=self.pcd_tree.search_radius_vector_3d(pos,self.UAV_safe_radius)
            # d_list=r[-1]
            # if r[0]==0:
            #     #safe
            #     return False
            # else:
            #
            #     #collision
            #     print(d_list)
            #     return True
            r = self.pcd_tree.search_knn_vector_3d(pos, 1)
            d_min = np.linalg.norm(self.pcd.points[r[1][0]] - pos)
            if d_min <= self.UAV_safe_radius:
                return True, d_min
            else:
                return False, d_min

    def get_next_point(self, pos_now, pos_prev, external_obs_list, external_eta_list, external_input=False):

        if not external_input:  # traditonal
            obs_list = []
            r = self.pcd_tree.search_radius_vector_3d(pos_now, self.r0)
            indexes = r[1]
            for i in indexes:
                if len(obs_list) > 500:
                    break
                obs_list.append(self.total_point_list[i])

            force = np.zeros(3)
            for idx, obs in enumerate(obs_list):
                force += self.repulsion_one_only(pos_now, 1.1, obs)
            attract_force = self.attraction(pos_now, 0.5)
            # special case
            if not (np.linalg.norm(force[0:2]) < 1e-7 or np.linalg.norm(attract_force[0:2]) < 1e-7):
                if abs(cal_angle(-force[0:2], attract_force[0:2])) <= 5:
                    norm_force = get_norm_vec(attract_force[0:2]) * np.linalg.norm(attract_force) * 0.2
                    force[0:2] += norm_force
            rep_force = force
            force += attract_force
            force = force / np.linalg.norm(force)
            step_size = self.stepSize
            # print("rep force:{}, attract force:{}".format(np.linalg.norm(rep_force),
            # np.linalg.norm(attract_force)))
            if pos_prev[0] is None:
                pos_next = force * step_size + pos_now
            else:
                pos_next = force * step_size + pos_now
                _, _, _, _, pos_next = self.kinematicConstrant(pos_now, pos_prev, pos_next)
            return pos_next, step_size


        else:  # using net
            obs_list = external_obs_list
            eta_list = external_eta_list

        # force = self.get_total_force(pos_now, obs_list, eta_list)

        attraction_force = self.attraction(pos_now, epsilon=np.mean(eta_list[-self.attraction_dim:]))
        rep_force = np.zeros(3)
        for idx, obs in enumerate(obs_list):
            rep_force += self.repulsion_one_only_v2(pos_now, eta_list[idx], obs)

        if not (np.linalg.norm(rep_force[0:2]) < 1e-7 or np.linalg.norm(attraction_force[0:2]) < 1e-7):
            if abs(cal_angle(-rep_force[0:2], attraction_force[0:2])) <= 5:
                norm_force = get_norm_vec(attraction_force[0:2]) * np.linalg.norm(attraction_force) * 0.2
                rep_force[0:2] += norm_force

        total_foce = rep_force + attraction_force
        force = total_foce / np.linalg.norm(total_foce)

        step_size = np.mean(eta_list[-(self.step_size_dim + self.attraction_dim):-self.attraction_dim])

        if pos_prev[0] is None:
            pos_next = force * step_size + pos_now
        else:
            pos_next = force * step_size + pos_now
            _, _, _, _, pos_next = self.kinematicConstrant(pos_now, pos_prev, pos_next)

        return pos_next, step_size

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(
                np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2

        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif 0 < x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])

        return x1, gam1, xres, gamres, qNext

    def getReward(self, flag, qBefore, q, qNext, step_size, d_min):  # 计算reward函数

        # assert step_size<=0.3+1e-4

        distance1 = distanceCost(qNext, self.end)
        distance2 = distanceCost(self.start, self.end)
        flag_done = False
        if flag:
            reward = -3000
            flag_done = True
            return reward, flag_done, distance1, distance2

        else:
            assert d_min > 0.3
            if qBefore[0] is not None:
                d1 = q - qBefore
                d2 = qNext - q
                dz = abs(qNext[2] - q[2])
                if dz < 0.3:
                    reward_for_pitch_angle = dz * -3.3333
                else:

                    reward_for_pitch_angle = np.clip(-dz ** 2 * 10 / 3 - dz * 7 / 3, -5, 0)
                reward_for_Y_angle = cal_angle(d1[0:2], d2[0:2]) / 90 * -1
                if distance1 > self.threshold:
                    delta_d = distance1 - np.linalg.norm(self.end - q)
                    if delta_d < 0:

                        # reward = (-distance1 / distance2 )
                        reward = -delta_d / self.stepSize * 10
                    else:
                        reward = -10

                else:  # reaching
                    reward = 3000

                if reward == 3000:
                    return reward, flag_done, distance1, distance2
                else:
                    reward += (reward_for_pitch_angle
                               + reward_for_Y_angle
                               + -10 * math.exp(-15 * (d_min - 0.3)))
                    reward = np.clip(reward, -20, 10)
                    reward = reward * step_size / self.stepSize
                    return reward, flag_done, distance1, distance2
            else:

                if distance1 > self.threshold:
                    delta_d = distance1 - np.linalg.norm(self.end - q)
                    if delta_d < 0:

                        # reward = (-distance1 / distance2 )
                        reward = -delta_d / self.stepSize * 10
                    else:
                        reward = -10

                    reward += -10 * math.exp(-15 * (d_min - 0.3))
                    reward = np.clip(reward, -20, 10)
                    reward = reward * step_size / self.stepSize
                else:
                    reward = 3000

            return reward, flag_done, distance1, distance2

    def cal_rewards_for_path(self, path_to_show):
        path_to_show = np.array(path_to_show)
        if path_to_show[0][0] is None:
            path_to_show = path_to_show[1:]
        for i in range(path_to_show.shape[0] - 1):
            idx_to_delete = []
            for j in range(i + 1, path_to_show.shape[0]):

                if np.linalg.norm(path_to_show[i] - path_to_show[j]) < 1e-3:
                    idx_to_delete.append(j)
            path_to_show = np.delete(path_to_show, idx_to_delete, 0)
        # path_to_show=np.vstack((np.array([None,None,None]),path_to_show))

        plt.subplot(211)

        plt.plot(path_to_show[1:, 0], path_to_show[1:, 1], c='red', marker='o', linewidth=0.3)
        plt.subplot(212)
        plt.plot([i for i in range(path_to_show.shape[0])], path_to_show[:, 2])
        plt.show()
        path_to_show = np.vstack((np.array([None, None, None]), path_to_show))
        step_size = 0.5
        pos_prev = np.array([None for _ in range(3)])

        reward_list = []
        for i in range(1, path_to_show.shape[0] - 1):
            pos_next = path_to_show[i + 1]
            pos_now = path_to_show[i]
            pos_prev = path_to_show[i - 1]
            flag, D_MIN = self.check_collison(pos_next)
            reward, _, _, _ = self.getReward(flag, pos_prev, pos_now, pos_next, np.linalg.norm(pos_next - pos_now),
                                             D_MIN)
            if flag:
                print(pos_next, i)

            reward_list.append(reward)

        plt.plot([i for i in range(len(reward_list))], reward_list)
        plt.show()
        print('\n', np.sum(reward_list))

    def calc_safety(self, pos):
        # assert type(pos)==np.ndarray and pos.shape==[3,1],pos.shape
        ret = self.pcd_tree.search_radius_vector_3d(pos.reshape(3, 1), 3)
        idx = np.array(ret[1], int)
        cnt = 0
        r = []
        for id in idx:
            if self.pcd.points[id][1] <= pos[1]:
                continue
            else:
                r.append(id)
                cnt += 1
            if cnt >= 49: break
        d = 0
        for id in r:
            d += distanceCost(self.pcd.points[id], pos)
        return d / (cnt * 1.5)

    def get_reward2(self, pos):

        if len(self.temp_traj) < 5:
            r1 = 0
            r2 = 0
        else:
            r1 = max(apf_env.calc_bending(self.temp_traj) * -5, -5)
            r2 = apf_env.calc_dist(self.temp_traj) * -1
        r3 = self.calc_safety(pos) * -1
        print(r1, r2, r3)

    # -------main loop----------------
    def main(self):
        pos_now = self.start
        pos_prev = np.array([None, None, None])
        for _ in tqdm(range(2000)):
            pos_next, _ = self.get_next_point(pos_now, pos_prev, None, None, False)
            if len(self.temp_traj) < 5:
                self.temp_traj.append(pos_next)
            else:
                self.temp_traj.pop(0)
                self.temp_traj.append(pos_next)

            ret = self.get_reward2(pos_next)
            self.path.append(pos_next)
            pos_prev = pos_now
            pos_now = pos_next
            if distanceCost(pos_next, self.end) < self.threshold:
                print("-----------------success in reaching goal------------------")
                break


class apf_env:
    def __init__(self, pcd_path, start, end):
        self.env_path = pcd_path
        print("________loading point cloud env from path: ", pcd_path)
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        # self.pcd=self.pcd.voxel_down_sample(voxel_size=0.5)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.step_size: float = None
        self.end = np.array(end)  # 目标点
        self.start = np.array(start)  # 轨迹起始点
        self.dgoal = 2.  # 当q与qgoal距离超过它时将衰减一部分引力
        self.r0 = 4  # interception range
        self.r1 = 1.
        self.threshold = 0.5  # q与qgoal距离小于它时终止训练或者仿真
        self.UAV_safe_radius = 0.3
        self.y_low_bound = 10
        self.y_high_bound = 30
        self.x_low_bound = -30
        self.x_high_bound = 50
        self.pcd_points = np.asarray(self.pcd.points)
        self.rep_max = -np.inf
        self.attraction_max = -np.inf
        self.path = [self.start.copy()]
        self.pos_now = self.start.copy()
        self.observation_now = None
        self.obs_points_num = -1
        self.vox_map: o3d.geometry.VoxelGrid = None
        self.center_coordinates = None
        self.temp_traj = [self.start]
        self.cnt = 0

    def reset(self):
        del self.path[1:]
        del self.temp_traj[1:]
        self.pos_now = self.start
        return self.get_observation(self.start)

    def get_observation(self, pos_now):
        ret = self.pcd_tree.search_radius_vector_3d(pos_now, self.r0)
        x, y, z = self.pos_now
        observation = [p for p in self.pcd_points[np.asarray(ret[1], int)] if  z-0.5<=p[2]<=z+0.5]
        self.obs_points_num = len(observation)

        if 0 < len(observation) <= 500:
            observation = np.asarray(observation)
            observation = np.vstack((observation, np.zeros([500 - observation.shape[0], 3])))
        elif len(observation) > 500:
            observation = np.array(observation[:500])
        else:
            observation = np.zeros([500, 3])
        assert observation.shape == (500, 3)
        if self.obs_points_num > 0:
            temp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observation[:self.obs_points_num, :]))
            voxel_local_map = o3d.geometry.VoxelGrid(o3d.geometry.VoxelGrid.create_from_point_cloud(temp_pcd, 0.5))
            self.vox_map = voxel_local_map
        else:
            self.vox_map = None
        self.observation_now = (np.array(observation), np.concatenate((pos_now, self.end)))

        return self.observation_now

    def get_next_point(self, pos_now, rep_coeff, attract_coeff, step_length):

        if not self.vox_map:
            rep_force = np.zeros(3)
            attraction_force = self.attraction(pos_now, attract_coeff)
        else:
            idxes = [v.grid_index for v in self.vox_map.get_voxels()]
            idxes.sort(key=lambda i: np.linalg.norm(pos_now - self.vox_map.get_voxel_center_coordinate(i)))

            center_coordinates = np.array([self.vox_map.get_voxel_center_coordinate(i) for i in idxes])
            self.center_coordinates = center_coordinates
            attraction_force = self.attraction(pos_now, attract_coeff)
            rep_force = np.zeros(3)

            for p, k in zip(center_coordinates[:50], rep_coeff):
                r = self.repulsion_one_only(pos_now, k, p)
                if math.hypot(*r)>=1e-5:
                    t = np.arccos(math.hypot(r[0],r[1])/math.hypot(r[0],r[1],r[2]))/np.pi*180
                    if t<10:
                        rep_force +=r

        if not (np.linalg.norm(rep_force[0:2]) < 1e-7 or np.linalg.norm(attraction_force[0:2]) < 1e-7):
            if abs(cal_angle(-rep_force[0:2], attraction_force[0:2])) <= 5:
                norm_force = get_norm_vec(attraction_force[0:2]) * np.linalg.norm(attraction_force) * 0.2
                rep_force[0:2] += norm_force

        total_foce = rep_force + attraction_force
        force = total_foce / np.linalg.norm(total_foce)*step_length
        pos_next = force  + pos_now
        # print("step_len={}".format(step_length))
        self.step_size = step_length
        pos_next[2] = np.clip(pos_next[2], 1.5, 2.15)
        if len(self.path)<2:
            return pos_next
        else:
            return self.kinamatic_constrain(pos_next)
    def kinamatic_constrain(self,pos_next):
        v1 = (self.path[-1]-self.path[-2])[:2]
        v2 = (pos_next-self.path[-1])[:2]
        d = math.hypot(*v2)
        angle = cal_angle(v1, v2)
        max_yaw = 30
        if angle <= max_yaw:
            return pos_next
        else:
            # to sphere_coordinate
            theta1 = math.atan2(v1[1], v1[0])
            theta2 = math.atan2(v2[1], v2[0])
            if theta1 >= 0 and theta2 >= 0:
                dtheta = theta2 - theta1
                if dtheta >= 0:
                    ret = np.array([d * np.cos(theta1 + max_yaw / 180 * np.pi), d * np.sin(theta1 + max_yaw / 180 * np.pi)])
                else:
                    ret = np.array([d * np.cos(theta1 - max_yaw / 180 * np.pi), d * np.sin(theta1 - max_yaw / 180 * np.pi)])
            elif theta1 >= 0 and theta2 < 0:
                if -theta2 + theta1 < np.pi - (-theta2 + theta1):
                    ret = np.array([d * np.cos(theta1 - max_yaw / 180 * np.pi), d * np.sin(theta1 - max_yaw / 180 * np.pi)])
                else:
                    ret = np.array([d * np.cos(theta1 + max_yaw / 180 * np.pi), d * np.sin(theta1 + max_yaw / 180 * np.pi)])

            elif theta1 < 0 and theta2 < 0:
                dtheta = theta2 - theta1
                if dtheta >= 0:
                    ret = np.array([d * np.cos(theta1 + max_yaw / 180 * np.pi), d * np.sin(theta1 + max_yaw / 180 * np.pi)])
                else:
                    ret = np.array([d * np.cos(theta1 - max_yaw / 180 * np.pi), d * np.sin(theta1 - max_yaw / 180 * np.pi)])
            elif theta1 < 0 and theta2 >= 0:
                if -theta1 + theta2 < np.pi - (-theta1 + theta2):
                    ret = np.array([d * np.cos(theta1 + max_yaw / 180 * np.pi), d * np.sin(theta1 + max_yaw / 180 * np.pi)])
                else:
                    ret = np.array([d * np.cos(theta1 - max_yaw / 180 * np.pi), d * np.sin(theta1 - max_yaw / 180 * np.pi)])

            return np.hstack((ret+self.path[-1][:2],pos_next[2]))

    def check_collision(self, pos_now):
        return self.vox_map.check_if_included(o3d.utility.Vector3dVector(pos_now.reshape(1, 3)))

    def attraction(self, pos_now, epsilon):  # 计算引力的函数
        epsilon = np.clip(epsilon, 0.8, 1.5)
        dist = distanceCost(pos_now, self.end)
        vec = self.end - pos_now
        if dist <= self.dgoal:
            f = epsilon * vec
        else:
            f = epsilon * self.dgoal * vec / dist
        # self.attraction_max = max(np.linalg.norm(f), self.attraction_max)
        return np.array(f)

    @staticmethod
    def calc_bending(waypts):
        ret = 0
        for i in range(len(waypts) - 2):
            v1 = waypts[i + 1][:2] - waypts[i][:2]
            v2 = waypts[i + 2][:2] - waypts[i + 1][:2]
            ret += cal_angle(v1, v2) / 60
        ret = ret / (len(waypts) - 2)
        # assert ret < 1.4,(ret,waypts)
        return ret

    @staticmethod
    def calc_dist(waypts):
        ret = 0
        line_dist = distanceCost(waypts[0], waypts[-1])
        for i in range(len(waypts) - 1):
            ret += distanceCost(waypts[i + 1], waypts[i])
        return ret / line_dist

    def calc_safety(self, pt):
        d = 0
        for pt_center in self.center_coordinates[:10]:
            d += distanceCost(pt, pt_center)
        d = d / (10 * 1.5)
        return d

    def repulsion_one_only(self, pos_now: np.ndarray, eta, obs_pos: np.ndarray):
        dist = distanceCost(pos_now, obs_pos)
        diff_vec = get_diff_vec(pos_now, obs_pos)
        if dist <= self.r1:
            f = eta * (1 / dist - 1 / self.r1) * (dist / self.r1) ** 2 * diff_vec + eta * (
                    1 / dist - 1 / self.r1) ** 2 * dist * diff_vec

            f = np.array(f)
        else:
            f = np.zeros(3)
        self.rep_max = max(self.rep_max, np.linalg.norm(f))
        return f

    def get_reward(self, pos, flag):
        # r1 theta and bending
        # r2 safe d

        if flag:
            return -100
        else:
            if len(self.temp_traj) < 5:
                r1 = -1.5
                r2 = -1.5
            else:
                r1 = max(self.calc_bending(self.temp_traj) * -2.5, -5)
                r2 = self.calc_dist(self.temp_traj) * -1
            r3 = self.calc_safety(pos) * -5
            d = distanceCost(pos, self.end) - distanceCost(self.path[-2], self.end)
            r4 = -d / self.step_size * 3.5
            r5 = 100 if distanceCost(pos, self.end) < 1. else 0
        return (r1 + r2 + r3 + 20) / 13 + r4 + r5

    def step(self, action):
        rep = (action[:-2] + 7/5) / 0.8  # 0.5 3
        attract = (action[-2] + 2) / 2  # 0.5,1.5
        step_length = (action[-1] + 5) / 20  # 0.2 0.3
        pos_next = self.get_next_point(self.pos_now, rep, attract, step_length)
        self.cnt += 1
        if len(self.temp_traj) < 5:
            self.temp_traj.append(pos_next)
        else:
            self.temp_traj.pop(0)
            self.temp_traj.append(pos_next)
            assert len(self.temp_traj) == 5
        self.path.append(pos_next)
        self.pos_now = pos_next
        flag2 = False
        if not (self.x_low_bound <= pos_next[0] <= self.x_high_bound and self.y_low_bound <= pos_next[
            1] <= self.y_high_bound and 1. <= pos_next[2] <= 2.2):
            flag2 = True
        flag = self.check_collision(pos_next)[0] or flag2
        if flag:
            print("INVALID")
        reward = self.get_reward(pos_next, flag)

        obs_ = self.get_observation(pos_next)

        return (obs_, reward, flag)


if __name__ == "__main__":

    attraction_eta_dim = 50
    step_size_eta_dim = 50
    root_path = r"/home/hit312/cyd/tiny-one/rollout_21-02-06_22-30-52/"
    traj_path = os.path.join(root_path, "reference_trajectory.csv")
    env_path = os.path.join(root_path, "pointcloud-unity.ply")
    try:
        path_exist = pd.read_csv(os.path.join(root_path, "path_26070.csv"), sep=',', header=None)
        path_exist = path_exist.to_numpy(dtype=float)
    except:
        path_exist = []
    try:
        reference_traj = pd.read_csv(
            traj_path,
            sep=',', header=None)
        reference_traj = reference_traj[1:].to_numpy(dtype=float)
        positon = reference_traj[1:, 1:4]
        velocity = reference_traj[1:, 4:7]
        acceleration = reference_traj[1:, 7:10]
        attitude = reference_traj[1:, 16:20]
        omega = reference_traj[1:, 21:24]
    except:
        reference_traj = []
    tmp = np.array([-8.5, 21., 2.3])
    apf = Traditional_Apf_Method(env_path=env_path, start=positon[0], end=positon[-1], step_size_dim=step_size_eta_dim,
                                 attraction_dim=attraction_eta_dim)
    # apf = Traditional_Apf_Method(env_path=r"/home/hit312/cyd/tmp/simple1.ply", start=np.array([-10,22,2]),
    #                 end=np.array([25,22,2]), step_size_dim=step_size_eta_dim,
    #                             attraction_dim=attraction_eta_dim)
    apf.main()
    # np.savetxt(env_path+r"\..\path.csv",apf.path,delimiter=',')
    # ---------------visualize--------------:
    visualize_and_collison_check_0(apf.path, apf.pcd)
    # apf.cal_rewards_for_path(apf.path)
