"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 04/11/2019
Updated: 07/11/2019
"""
import os
import numpy as np
import cv2

path = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/dataset_collector/data"
time = "20191107-172035"

rgb = os.path.join(path, "rgb", "rgb{0}.jpeg".format(time))
bb_3d_file = os.path.join(path, "bbox", "bb{0}.npz".format(time))
bb_coordinate_file = os.path.join(path, "bbox", "bb_coord{0}.npz".format(time))
semantic_file = os.path.join(path, "semantic", "semantic{0}.npz".format(time))
depth_file = os.path.join(path, "depth", "depth{0}.npy".format(time))
sensor_width = 1024
sensor_height = 768


def process_bb(bb_data_numpy):
    valid_bbs = []
    for bb_3d in bb_data_numpy:
        bb_2d = [x[:-1] for x in bb_3d]  # Excluding Z axis information since the depth was already checked before
        object_bbs = []
        for bb in bb_2d:
            bb = [int(x) for x in bb]
            # Checking if there are bboxes out of the image view
            if 0 <= bb[0] <= sensor_width and 0 <= bb[1] <= sensor_height:
                object_bbs.append(bb[0])
                object_bbs.append(bb[1])

        # Finds the maximum possible size to a 2D bounding box
        if object_bbs:
            even_values = object_bbs[::2]
            odd_values = object_bbs[1::2]
            x_min = min(even_values)
            y_min = min(odd_values)
            x_max = max(even_values)
            y_max = max(odd_values)
            shape = (x_max-x_min, y_max-y_min)
            size = shape[0]*shape[1]
            proportion = size/(sensor_height*sensor_width)
            # print('shape {0}; size {1}; proportion {2}'.format(shape, size, proportion))
            if proportion > 1e-4:  # 1e-4 was a value found by observation
                valid_bbs.append([x_min, y_min, x_max, y_max])
    return valid_bbs


def process_semantic_data(semantic_numpy):
    # codification: https://carla.readthedocs.io/en/latest/cameras_and_sensors/#sensorcamerasemantic_segmentation
    # red_codification = {0: "Unlabeled", 4: "Pedestrian", 10: "Car"}
    semantic_numpy[(semantic_numpy != 10) & (semantic_numpy != 4)] = 0  # Background
    semantic_numpy[semantic_numpy == 4] = 127  # Pedestrian
    semantic_numpy[semantic_numpy == 10] = 255  # Car
    cv2.imwrite('semantic.png', semantic_numpy)
    return semantic_numpy


def filter_bb_by_semantic_data(bounding_boxes, semantic_img, class_codification):
    """
    - On BB area multiple classes can exist simultaneously on the semantic notation, and that's ok
    - [UPDATE: Didn't work out, since the bboxes aren't tight enough to the object. Thus some checked areas kept being
     marked as unchecked...] Use semantic information to EXCLUDE frames where obvious cars are not detected (see if car exists in semantic
    image but is not apparent on rgb with bounding boxes) Clone the semantic image, and as each bbox goes through the
     image, paint the referred area as black. At the end, a simple check if there are any not black pixels on the cloned
     semantic img would be used to check if the image should be discarded or not.
     
    """

    valid_bounding_boxes = []
    for bb in bounding_boxes:
        x_min, y_min, x_max, y_max = bb
        img_roi = semantic_img[y_min:y_max, x_min:x_max]
        # Check on semantic img's class pixels if at least one pixel of the given class exists on the roi area
        if (img_roi[img_roi == class_codification]).any():
            valid_bounding_boxes.append(bb)
    return valid_bounding_boxes


def proccess_depth(depth_file):
    data = np.load(depth_file)

    data = data.reshape((768, 1024, 4))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    return depth_meters


def compare_depth_and_bb_coordinate(depth_array, bb_3d_actor, bb_coordinate_actor):
    # The array of [x, y, z] of bounding box coordinates from each bounding box is converted to [pixel, pixel, meters]
    # by using the bb_3d [x, y] which are already [pix, pix] and the bb_coordinate [z] which is [meters]. This is
    # possible to be done since bb_3d and bb_coordinate were captured at the same time and assigned in the list in the
    # same order, and thus point to the same actor in the world
    bb_new_coordinate_actor = bb_3d_actor[:]

    for walker_id in range(len(bb_new_coordinate_actor)):
        walker_coordinate = bb_coordinate_actor[walker_id]
        for box_point in range(len(bb_new_coordinate_actor[walker_id])):
            bb_new_coordinate_actor[walker_id][box_point][2] = walker_coordinate[2][box_point]

    # Adjusting/removing the box coordinates which do not fit the image shape
    bb_new_coordinate_actor = fit_boxes_to_img_shape(bb_new_coordinate_actor, sensor_width, sensor_height)

    # Comparing each bb z point to the respective depth array
    corrected_bb = bb_new_coordinate_actor[:]
    for actor_idx, actor in enumerate(bb_new_coordinate_actor):
        for bb_idx, bb in enumerate(actor):
            if bb[0] != -1 and bb[1] != -1:  # x and y are set to -1 to flag invalid points
                depth_sensor_value = depth_array[int(bb[1])][int(bb[0])]
                box_z_point = bb[2]
                # If the 3d box is occluded then we invalidate this set of points
                # FIXME # Changing the lower bound value of this condition makes more cars appear...
                #  Maybe the depth information is not that accurate from the bounding box. Check the other coordinate
                #  system from client_bounding_boxes.py that is y inverse or something like that !!!
                if not(0 <= box_z_point <= depth_sensor_value):
                    corrected_bb[actor_idx][bb_idx] = np.array([-1, -1, -1])
    return corrected_bb


def fit_boxes_to_img_shape(bb_new_coordinate_actor, sensor_width, sensor_height):
    # x and y are set to -1 to flag invalid points
    adequate_boxes = -np.ones(bb_new_coordinate_actor.shape)
    for actor_index, actor in enumerate(bb_new_coordinate_actor):
        actor_bbs = -np.ones(bb_new_coordinate_actor[0].shape)
        for bb_index, bb in enumerate(actor):
            if 0 <= bb[0] <= sensor_width and 0 <= bb[1] <= sensor_height:
                actor_bbs[bb_index] = np.array([int(bb[0]), int(bb[1]), bb[2]])
        adequate_boxes[actor_index] = actor_bbs
    return adequate_boxes


if __name__ == "__main__":
    # Bounding Box processing
    bb_3d_data = np.load(bb_3d_file)
    bb_3d_vehicles = bb_3d_data['arr_0']
    bb_3d_walkers = bb_3d_data['arr_1']
    # valid_bb_vehicles = process_bb(bb_vehicles)
    # valid_bb_walkers = process_bb(bb_walkers)

    # Depth processing
    depth_array = proccess_depth(depth_file)

    # Depth + bb coordinate check
    bb_coordinate_data = np.load(bb_coordinate_file)
    bb_coordinate_vehicles = bb_coordinate_data['arr_0']
    bb_coordinate_walkers = bb_coordinate_data['arr_1']

    vehicles_bb = compare_depth_and_bb_coordinate(depth_array, bb_3d_vehicles, bb_coordinate_vehicles)
    walkers_bb = compare_depth_and_bb_coordinate(depth_array, bb_3d_walkers, bb_coordinate_walkers)

    valid_bb_vehicles = process_bb(vehicles_bb)
    valid_bb_walkers = process_bb(walkers_bb)


    # # Semantic processing
    # semantic_img = process_semantic_data(np.load(semantic_file)['arr_0'])
    #
    # # Filtering bounding boxes according to semantic annotations
    # valid_bb_vehicles_2 = filter_bb_by_semantic_data(valid_bb_vehicles, semantic_img, class_codification=np.array([0, 0, 255]))
    # valid_bb_walkers_2 = filter_bb_by_semantic_data(valid_bb_walkers, semantic_img, class_codification=np.array([0, 0, 127]))

    # RGB processing
    rgb_img = cv2.imread(rgb)
    cv2.imwrite('raw_img.png', rgb_img)

    # Bounding boxes without segmentation filter
    for bb in valid_bb_vehicles:
        cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
    for bb in valid_bb_walkers:
        cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

    # # Bounding boxes with segmentation filter
    # rgb_img_2 = cv2.imread(rgb)
    # for bb in valid_bb_vehicles_2:
    #     cv2.rectangle(rgb_img_2, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
    # for bb in valid_bb_walkers_2:
    #     cv2.rectangle(rgb_img_2, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 0), 1)

    cv2.imwrite('filtered_boxed_img.png', rgb_img)
    print('bb_walkers', len(valid_bb_walkers))
    print('bb_vehicles', len(valid_bb_vehicles))
    # cv2.imwrite('boxed_img.png', rgb_img_2)
    # print('filtered_bb_walkers', len(valid_bb_walkers_2))
    # print('filtered_bb_vehicles', len(valid_bb_vehicles_2))
