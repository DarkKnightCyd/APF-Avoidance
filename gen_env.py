import math
import numpy as np
import open3d as o3d
import os
import pandas as pd
import copy
import random
import itertools
import math
from itertools import permutations
def gen_sign(pos_xy,s=True):
    pts = np.hstack((np.tile(pos_xy,(50,1)),np.linspace(-1,5,50).reshape(-1,1)))
    temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if s:
        temp.paint_uniform_color([4/255.,186/255.,21/255.])
    else:
        temp.paint_uniform_color([240 / 255., 175 / 255., 247 / 255.])
    return temp
def gen_tree(offset_pos):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tree_points+offset_pos))

def generate_cubes(radius, centre_pos, height):
    cube_points_horizon=[]
    dx = 0.05
    N = math.floor(radius / dx)
    for i in range(-N,N+1):
       for j in range(-N,N+1):
            cube_points_horizon.append(centre_pos+np.array([i*dx,j*dx]))
    cube_points_horizon = np.array(cube_points_horizon)
    dz = 0.1
    z_list = [-0.5 + i * dz for i in range(math.ceil(height / dz))]
    cube_points_total=[]
    for z in z_list:
        zz = np.array([z] * cube_points_horizon.shape[0])
        zz = np.expand_dims(zz, 1)
        cube_points_total.append(np.hstack((cube_points_horizon, zz)))
    rp = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.array(cube_points_total).reshape(-1, 3))
    )
    # rp.paint_uniform_color([255. / 255., 217. / 255., 102. / 255.])
    return rp

def generate_spheres(centre_pos,radius):
    dtheta = np.radians(0.5)
    ret = []
    z = np.linspace(-radius,radius,100)
    for z_now in z:
        radius_now = np.sqrt(radius ** 2 - z_now ** 2)
        for i in range(int(np.floor(2*np.pi/dtheta))):
            x = radius_now*np.cos(i*dtheta)
            y = radius_now*np.sin(i*dtheta)
            ret.append(np.array([x,y,z_now]))
    rp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.asarray(ret)+np.array([*centre_pos,random.uniform(0.4,2)])
    ))
    # rp.paint_uniform_color([255. / 255., 217. / 255., 102. / 255.])
    return rp

def generate_columns(radius,centre_pos,height):

    z = np.linspace(0,height,100)
    dtheta = np.radians(0.5)
    ret = []
    for z_now in z:
        for i in range(int(np.floor(2 * np.pi / dtheta))):
            x = radius * np.cos(i * dtheta)
            y = radius * np.sin(i * dtheta)
            ret.append(np.array([x, y, z_now]))
    rp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(ret)+np.array([*centre_pos,0])))
    # rp.paint_uniform_color([255./255.,217./255.,102./255.])
    return rp

if __name__=='__main__':

    root_path = r"D:/pointcloud_base"
    # base_tree = pd.read_csv(os.path.join(os.path.dirname(__file__),"base_tree.csv")
    #         ,sep=',',header=None).to_numpy()
    for root,dir,files in os.walk(root_path,topdown=True,followlinks=True):
        files_set=files

    base_env=files_set[random.randint(0,len(files_set))]
    pcd=o3d.io.read_point_cloud(os.path.join(root_path,base_env))
    #o3d.visualization.draw_geometries([pcd])
    x_list = np.linspace(-30,30,500)
    y_list = np.linspace(5,35,200)

    pt_ground_xy = np.array(list(itertools.product(x_list,y_list)))
    l1= pt_ground_xy.shape[0]
    points_for_ground = np.array([np.hstack((pt_ground_xy, np.array([-0.1+i*0.1]*l1).reshape(l1,1))) for i in range(7)]).reshape(-1,3)

    # for p in pcd.points:
    #     if p[2]<=0.4:
    #         points_for_ground.append(p)
    points_for_ground = np.array(points_for_ground)
    base_env=o3d.open3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_for_ground))
    base_env.paint_uniform_color([9/255.,108/255.,188/255.])
    #base_env.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,10,255]),(points_for_ground.shape[0],1)))

    # o3d.visualization.draw_geometries([base_env])
    x_boundary = np.array([-20,30])
    y_boundary = np.array([12,28])
    center_pos =np.array([
        [random.uniform(x_boundary[0],x_boundary[1]),
        random.uniform(y_boundary[0],y_boundary[1])] for i in range(15)
    ])
    obs_list = []
    obs_info = []
    for i,center_pt in enumerate(center_pos):

        radius = random.uniform(5,10)

        u = random.uniform(0,1)
        if u<=0.5:
            gen_center_pt = np.random.multivariate_normal(mean=center_pt,
                                                          cov=np.array([[25, 0],
                                                                        [0, 25]]),
                                                          size= 2
                                                          )
            ret = []
            for x in gen_center_pt:
                rd = random.uniform(0.3, 0.7)
                c_pos = x
                h = random.uniform(4, 7)
                ret.append(generate_columns(rd, c_pos, h))
                obs_info.append({'type': 'Cylinder', 'radius': rd, 'c_p': x, 'h': h})
            #ret = [generate_columns(random.uniform(0.3, 0.7), x, random.uniform(4, 7)) for x in gen_center_pt]
        # ret = [generate_spheres(x,random.uniform(0.4,1.4)) for x in gen_center_pt]
        # ret = [generate_cubes(pcd,random.uniform(0.4,1),x,random.uniform(2,7)) for x in gen_center_pt]
        elif u< -1:
            gen_center_pt = np.random.multivariate_normal(mean=center_pt,
                                                          cov=np.array([[16, 0],
                                                                        [0, 16]]),
                                                          size= 2
                                                          )
            ret = [generate_spheres(x, random.uniform(0.4, 1.4)) for x in gen_center_pt]
        else:
            gen_center_pt = np.random.multivariate_normal(mean=center_pt,
                                                          cov=np.array([[16, 0],
                                                                        [0, 16]]),
                                                          size=3
                                                          )
            ret = []
            for x in gen_center_pt:
                c_pos = x
                r = random.uniform(0.4, 1.4)
                h = random.uniform(2, 7)
                ret.append(generate_cubes(r, c_pos, h))
                obs_info.append({'type': 'cube', 'radius': r, 'c_p': x, 'h': h})
            #ret = [generate_cubes(random.uniform(0.4, 1), x, random.uniform(2, 7)) for x in gen_center_pt]

        obs_list+=ret




    o3d.visualization.draw_geometries([base_env,*obs_list])
    p_o = [np.asarray(o.points) for o in obs_list]
    points = np.vstack((base_env.points,*p_o))
    o3d.io.write_point_cloud("./pcd/t6.ply",o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points)
    ))
    print("f")
    #o3d.io.write_point_cloud(r"/home/hit312/cyd/tmp/simple1.ply",base_env)



