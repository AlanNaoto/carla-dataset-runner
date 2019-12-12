import numbers
import h5py
import cv2
import numpy as np
import sys
from check_for_n_occluded_points import get_4_points_max_2d_area, adjust_points_to_img_size, \
    tighten_bbox_points, compute_bb_coords


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


def filter_bounding_boxes(rgb_data, bb_data, depth_data, actor):
    good_bounding_boxes = []
    depth_data = np.transpose(depth_data)
    frame_width = 1024
    frame_height = 768

    assert actor=="vehicle" or actor=="walker"
    if actor == "vehicle":
        bounds = [-frame_width/4, frame_width*5/4, -frame_height/4, frame_height*5/4]
    elif actor == "walker":
        bounds = [0, frame_width, 0, frame_height]

    for actor_bb_3d in bb_data:
        # Apply some medium constraining on the data to not exclude every impossible point
        possible_bb_3d_points = np.array([x for x in actor_bb_3d if 
                                        bounds[0] <= x[0] <= bounds[1] and bounds[2] <= x[1] <= bounds[3]])
        if len(possible_bb_3d_points) < 2:  # You can't have a box with only one point!
            continue
        # Transform out of boundaries points into possible points
        possible_bb_3d_points = adjust_points_to_img_size(frame_width, frame_height, possible_bb_3d_points)
        possible_bb_3d_points, bbox_exists, max_2d_area = get_4_points_max_2d_area(possible_bb_3d_points)

        if bbox_exists:
            xmin, ymin, xmax, ymax, visible_points = tighten_bbox_points(possible_bb_3d_points, depth_data)
            if all([isinstance(x, numbers.Number) for x in [xmin, ymin, xmax, ymax]]):
                tightened_bb_area = (xmax - xmin) * (ymax - ymin)
                tightened_bb_proportion = tightened_bb_area/max_2d_area
                tightened_bb_size_to_img = tightened_bb_area/(frame_height*frame_width)
                if tightened_bb_size_to_img > 2.5E-4:
                    # colormap = {'3or4': (0, 255, 0), '2': (255, 0, 0), "1": (0, 0, 255), "0": (255, 255, 255)}
                    # cv2.rectangle(rgb_data, (xmin, ymin), (xmax, ymax), colormap[visible_points], 1)
                    good_bounding_boxes.append([xmin, ymin, xmax, ymax, visible_points])

    # Check if there is too much intersection over union between bounding boxes
    good_bounding_boxes = remove_bbs_too_much_IOU(good_bounding_boxes)
    return good_bounding_boxes


def remove_bbs_too_much_IOU(bounding_boxes):
    bounding_boxes = np.array([x[:-1] for x in bounding_boxes])  # Removing the color index
    # If two bbs are overlapping too much, then we make a new bbox which takes the max size of the 
    # union of both boxes
    if len(bounding_boxes) > 2:
        there_are_overlapping_boxes = True
        while there_are_overlapping_boxes:
            there_are_overlapping_boxes = False
            bb_idx = 0
            while bb_idx < len(bounding_boxes):
                bb_ref = bounding_boxes[bb_idx]
                bb_compared_idx = bb_idx + 1
                while bb_compared_idx < len(bounding_boxes):
                    bb_compared = bounding_boxes[bb_compared_idx]
                    # Compute intersection - Min of the maxes; max of the mins
                    xmax = min(bb_ref[2], bb_compared[2])
                    xmin = max(bb_ref[0], bb_compared[0])
                    ymin = max(bb_ref[1], bb_compared[1])
                    ymax = min(bb_ref[3], bb_compared[3])
                    # Check if there is intersection between the bbs
                    if (xmax-xmin) > 0 and (ymax-ymin) > 0:
                        intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        bb_ref_area = (bb_ref[2] - bb_ref[0] + 1) * (bb_ref[3] - bb_ref[1] + 1)
                        bb_compared_area = (bb_compared[2] - bb_compared[0] + 1) * (bb_compared[3] - bb_compared[1] + 1)
                        IoU = intersection_area / (bb_compared_area + bb_ref_area - intersection_area)
                        if IoU > 0.90:
                            there_are_overlapping_boxes = True
                            xmin = min(bb_compared[0], bb_ref[0])
                            ymin = min(bb_compared[1], bb_ref[1])
                            xmax = max(bb_compared[2], bb_ref[2])
                            ymax = max(bb_compared[3], bb_ref[3])
                            bounding_boxes[bb_idx] = [xmin, ymin, xmax, ymax]
                            bounding_boxes = np.delete(bounding_boxes, (bb_compared_idx), axis=0)

                    bb_compared_idx += 1
                bb_idx += 1

    return bounding_boxes


if __name__ == "__main__":
    hdf5_file = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/carla-dataset-runner/data/town02.hdf5"
    # rgb_data, bb_data_vehicles, bb_data_walkers, depth_data = read_hdf5_test(hdf5_file)
    # treat_single_image(rgb_data, bb_data_vehicles, bb_data_walkers, depth_data, save_to_many_single_files=True)
    create_video_sample(hdf5_file, 'town02', show_depth=False)



