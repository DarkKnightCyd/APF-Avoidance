import math
import numpy as np
import open3d as o3d
import os
import pandas as pd
import copy
import random
import itertools
import math
def gen_sign(pos_xy,s=True):
    pts = np.hstack((np.tile(pos_xy,(10,1)),np.linspace(-1,5,10).reshape(-1,1)))
    temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if s:
        temp.paint_uniform_color([68/255.,13/255.,177/255.])
    else:
        temp.paint_uniform_color([240 / 255., 175 / 255., 247 / 255.])
    return temp

if __name__ =='__main__':
    pcd = o3d.io.read_point_cloud("./pcd/t2.ply")
    # o3d.visualization.draw_geometries([pcd])
    start = []
    for i in range(5):
        s_now = random.uniform()