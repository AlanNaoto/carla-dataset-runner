import h5py
import cv2
import numpy as np
import sys


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

    # bb
    bb_vehicles = bb_vehicles_data
    bb_walkers = bb_walkers_data

    if all(bb_vehicles != -1):
        for bb_idx in range(0, len(bb_vehicles), 4):
            coordiante_min = (int(bb_vehicles[0 + bb_idx]), int(bb_vehicles[1 + bb_idx]))
            coordinate_max = (int(bb_vehicles[2 + bb_idx]), int(bb_vehicles[3 + bb_idx]))
            cv2.rectangle(rgb_data, coordiante_min, coordinate_max, (0, 255, 0), 1)
    if all(bb_walkers != -1):
        for bb_idx in range(0, len(bb_walkers), 4):
            coordiante_min = (int(bb_walkers[0 + bb_idx]), int(bb_walkers[1 + bb_idx]))
            coordinate_max = (int(bb_walkers[2 + bb_idx]), int(bb_walkers[3 + bb_idx]))

            cv2.rectangle(rgb_data, coordiante_min, coordinate_max, (0, 0, 255), 1)
    if save_to_many_single_files:
        cv2.imwrite('filtered_boxed_img.jpeg', rgb_data)

    # depth
    normalized_depth = cv2.normalize(depth_data, depth_data, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # normalized_depth = np.stack((normalized_depth,)*3, axis=-1)  # Grayscale into 3 channels
    normalized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_HOT)
    if save_to_many_single_files:
        cv2.imwrite('depth_minmaxnorm.png', normalized_depth)
    return rgb_data, normalized_depth


def create_video_sample(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = file.attrs['sensor_width']
        frame_height = file.attrs['sensor_height']
        # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width*2, frame_height))
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (frame_width * 2, frame_height))
        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            rgb_data = np.array(file['rgb'][str(time)])
            bb_vehicles_data = np.array(file['bounding_box']['vehicles'][str(time)])
            bb_walkers_data = np.array(file['bounding_box']['walkers'][str(time)])
            depth_data = np.array(file['depth'][str(time)])

            sys.stdout.write("\r")
            sys.stdout.write('Recording video. Frame {0}/{1}'.format(time_idx, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()

            rgb_frame, depth_frame = treat_single_image(rgb_data, bb_vehicles_data, bb_walkers_data, depth_data)
            composed_frame = np.hstack((rgb_frame, depth_frame))
            out.write(composed_frame)
    print('\nDone.')

if __name__ == "__main__":
    # rgb_data, bb_data_vehicles, bb_data_walkers, depth_data = read_hdf5_test("/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/dataset_collector/data/carla_dataset.hdf5")
    # treat_single_image(rgb_data, bb_vehicles_data, bb_walkers_data, save_to_many_single_files=True)
    create_video_sample("/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/dataset_collector/data/carla_dataset.hdf5")



