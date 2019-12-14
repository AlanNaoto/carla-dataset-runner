import numbers
import h5py
import cv2
import numpy as np
import sys
from check_for_n_occluded_points import get_4_points_max_2d_area, adjust_points_to_img_size, \
    tighten_bbox_points, compute_bb_coords, filter_bounding_boxes


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


def create_video_sample(hdf5_file, video_name, show_depth=True):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = file.attrs['sensor_width']
        frame_height = file.attrs['sensor_height']
        if show_depth:
            frame_width = frame_width * 2
        out = cv2.VideoWriter(video_name+'.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (frame_width, frame_height))

        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            # HDF5 reading
            rgb_data = np.array(file['rgb'][str(time)])
            bb_vehicles_data = np.array(file['bounding_box']['vehicles'][str(time)])
            bb_walkers_data = np.array(file['bounding_box']['walkers'][str(time)])
            depth_data = np.array(file['depth'][str(time)])

            sys.stdout.write("\r")
            sys.stdout.write('Recording video. Frame {0}/{1}'.format(time_idx, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()

            # Bounding boxes
            vehicle_bbs = filter_bounding_boxes(rgb_data, bb_vehicles_data, depth_data, 'vehicle')
            walker_bbs = filter_bounding_boxes(rgb_data, bb_walkers_data, depth_data, 'walker')

            # Editing image
            # colormap = {'3or4': (0, 255, 0), '2': (255, 0, 0), "1": (0, 0, 255), "0": (255, 255, 255)}
            [cv2.rectangle(rgb_data, (x[0], x[1]), (x[2], x[3]), (0, 255, 0), 1) for x in vehicle_bbs]
            [cv2.rectangle(rgb_data, (x[0], x[1]), (x[2], x[3]), (0, 0, 255), 1) for x in walker_bbs]
            
            cv2.putText(rgb_data, 'timestamp', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(rgb_data, str(time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(rgb_data)

    print('\nDone.')


if __name__ == "__main__":
    hdf5_file = "/media/alan/Seagate Expansion Drive/Data/CARLA/raw_+bbs/town01.hdf5"
    # rgb_data, bb_data_vehicles, bb_data_walkers, depth_data = read_hdf5_test(hdf5_file)
    # treat_single_image(rgb_data, bb_data_vehicles, bb_data_walkers, depth_data, save_to_many_single_files=True)
    create_video_sample(hdf5_file, 'town01', show_depth=False)



