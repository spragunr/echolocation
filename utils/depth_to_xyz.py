"""
Utility code for converting from depth maps to point clouds.
Logic is borrowed from:

https://github.com/ros-perception/image_pipeline/
blob/indigo/depth_image_proc/include/depth_image_proc/depth_conversions.h

"""
import sys
import yaml
import numpy as np


def parse_calibration_yaml(calib_file):
    """Read in a .yaml file in ROS format with camera parameters. Returns
    a dictionary.

    """
    with open(calib_file, 'r') as f:
        params = yaml.load(f)
        return params


def depth_map_to_point_cloud_slow(depth_map, cam_info):
    """Takes in a depth map of size (height, width) and returns a point
    cloud of size (at most) (height*width , 3).

    This is a slow version that is pretty much a direct translation of
    a ROS nodelet that does the same thing. This shouldn't be used,
    but I'm leaving it here to illustrate how the process works.

    """
    points = []
    P = cam_info['projection_matrix']['data']
    scaling_factor = 1/1000.0
    center_x = P[2]
    center_y = P[6]
    constant_x = scaling_factor / P[0]
    constant_y = scaling_factor / P[5]
    print constant_x
    print constant_y

    for v in range(depth_map.shape[1]):
        for u in range(depth_map.shape[0]):
            if depth_map[u, v] != 0:
                X = (v - center_x) * depth_map[u, v] * constant_x
                Y = (u - center_y) * depth_map[u, v] * constant_y
                Z = depth_map[u, v] * scaling_factor
                points.append((X, Y, Z))
    return points

def depth_map_to_point_cloud(depth_map, cam_info):
    """Takes in a depth map of size (height, width) and returns a point
    cloud of size (at most) (height*width , 3).

    This is a fast vectorized version.

    """
    points = []
    P = cam_info['projection_matrix']['data']
    scaling_factor = 1/1000.0
    center_x = P[2]
    center_y = P[6]
    constant_x = scaling_factor / P[0]
    constant_y = scaling_factor / P[5]

    u = np.arange(0, depth_map.shape[0], 1)
    u = np.tile(np.array([u]).T, (1, depth_map.shape[1]))

    v = np.arange(0, depth_map.shape[1], 1)
    v = np.tile(v, (depth_map.shape[0], 1))

    full = np.stack([u, v, depth_map], axis=2)
    full = full.reshape(depth_map.shape[0] * depth_map.shape[1], -1)
    zeros = np.argwhere(full[:, 2] == 0)
    full = np.delete(full, zeros, axis=0)

    X = (full[:, 1] - center_x) * full[:, 2] * constant_x
    Y = (full[:, 0] - center_y) * full[:, 2] * constant_y
    Z = full[:, 2] * scaling_factor

    points = np.stack([X, Y, Z], axis=1)
    return points

def main():
    """ Demo """
    import h5py
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    cam_info = parse_calibration_yaml(sys.argv[1])

    with h5py.File(sys.argv[2], 'r') as d:
        for i in range(d['depth'].shape[0]):
            img = d['depth'][i, ...]
            pc = depth_map_to_point_cloud(img, cam_info)

            # plt.imshow(img)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], marker='s')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Z')
            # ax.set_zlabel('Y')
            # plt.show()
            print i

if __name__ == "__main__":
    main()
