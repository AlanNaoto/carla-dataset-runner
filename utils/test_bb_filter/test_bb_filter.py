import h5py
import cv2
import numpy as np
import sys
from check_for_n_occluded_points import get_bbox_for_2_visible_points, get_4_points_max_2d_area, \
    adjust_points_to_img_size, compute_bb_coords, get_bbox_for_1_visible_point


def read_hdf5_test(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        rgb = file['rgb']
        bb_vehicles = file['bounding_box']['vehicles']
        bb_walkers = file['bounding_box']['walkers']
        depth = file['depth']
        timestamps = file['timestamps']
        for time in timestamps['timestamps']:
            rgb_data = np.array(rgb[str(time)])
            bb_vehicles_data = np.array(bb_vehicles[str(time)])
            bb_walkers_data = np.array(bb_walkers[str(time)])
            depth_data = np.array(depth[str(time)])
            return rgb_data, bb_vehicles_data, bb_walkers_data, depth_data


def treat_single_image(rgb_data, bb_vehicles_data, bb_walkers_data, depth_data, save_to_many_single_files=False):
    # raw rgb
    if save_to_many_single_files:
        cv2.imwrite('raw_img.jpeg', rgb_data)

    depth_data = np.transpose(depth_data)
    frame_width = 1024
    frame_height = 768
    xmin, ymin, xmax, ymax = None, None, None, None
    for vehicle_bb_3d in bb_vehicles_data:
        # Apply some medium constraining on the data to not exclude every impossible point
        possible_bb_3d_points = np.array([x for x in vehicle_bb_3d if
                                          (0-frame_width/2 <= int(x[0]) <= frame_width+frame_width/2) and
                                          (0-frame_height/2 <= int(x[1]) <= frame_height+frame_height/2)])
        if len(possible_bb_3d_points) < 2:  # You can't have a box with only one point!
            continue
        # Transform out of boundaries points into possible points
        possible_bb_3d_points = adjust_points_to_img_size(frame_width, frame_height, possible_bb_3d_points)
        possible_bb_3d_points, bbox_exists = get_4_points_max_2d_area(possible_bb_3d_points)

        if bbox_exists:
            xmin, ymin, xmax, ymax = tighten_bbox_points(possible_bb_3d_points, depth_data)
            if (xmin and ymin and xmax and ymax):
                cv2.rectangle(rgb_data, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    if save_to_many_single_files:
        cv2.imwrite('filtered_boxed_img.png', rgb_data)

    return rgb_data


def check_visible_points(possible_bb_3d_points, depth_data):
    visible_points = []
    # Check which points are occluded
    for xyz_point in possible_bb_3d_points:
        x = int(xyz_point[0])
        y = int(xyz_point[1])
        z = xyz_point[2]
        depth_on_sensor = depth_data[x][y]
        point_visible = False
        if 0 <= z <= depth_on_sensor:
            point_visible = True
        visible_points.append(point_visible)
    return visible_points  # e.g.: [True, True, False, False]


def tighten_bbox_points(possible_bb_3d_points, depth_data):
    points_occlusion_status = check_visible_points(possible_bb_3d_points, depth_data)
    # No points with occlusion
    if points_occlusion_status.count(True) == 4 or points_occlusion_status.count(True) == 3:
        xmin, ymin, xmax, ymax = compute_bb_coords(possible_bb_3d_points)
        return xmin, ymin, xmax, ymax
    
    # A pair of points occluded
    elif points_occlusion_status.count(True) == 2:
        xmin, ymin, xmax, ymax = get_bbox_for_2_visible_points(possible_bb_3d_points, depth_data, points_occlusion_status)
        return xmin, ymin, xmax, ymax

    # FIXME maybe do some analysis on occluded area occupied
    elif points_occlusion_status.count(True) == 1:
        xmin, ymin, xmax, ymax = get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, points_occlusion_status)
        return None, None, None, None
    
    elif points_occlusion_status.count(True) == 0:
        return None, None, None, None


def create_video_sample(hdf5_file, show_depth=True):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = file.attrs['sensor_width']
        frame_height = file.attrs['sensor_height']
        if show_depth:
            frame_width = frame_width * 2
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1, (frame_width, frame_height))

        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            rgb_data = np.array(file['rgb'][str(time)])
            bb_vehicles_data = np.array(file['bounding_box']['vehicles'][str(time)])
            bb_walkers_data = np.array(file['bounding_box']['walkers'][str(time)])
            depth_data = np.array(file['depth'][str(time)])

            sys.stdout.write("\r")
            sys.stdout.write('Recording video. Frame {0}/{1}'.format(time_idx, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()
            rgb_frame = treat_single_image(rgb_data, bb_vehicles_data, bb_walkers_data, depth_data)
            cv2.putText(rgb_frame, 'timestamp', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(rgb_frame, str(time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(rgb_frame)

    print('\nDone.')


if __name__ == "__main__":
    hdf5_file = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/carla-dataset-runner/data/testing.hdf5"
    # rgb_data, bb_data_vehicles, bb_data_walkers, depth_data = read_hdf5_test(hdf5_file)
    # treat_single_image(rgb_data, bb_data_vehicles, bb_data_walkers, depth_data, save_to_many_single_files=True)
    create_video_sample(hdf5_file, show_depth=False)



