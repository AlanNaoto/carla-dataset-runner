"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 04/11/2019
Updated: 14/11/2019
"""
import cv2
import numpy as np


def apply_filters_to_3d_bb(bb_3d_data, depth_array, sensor_width, sensor_height, camera_x_location, camera_y_location,
                           camera_z_location):
    # Bounding Box processing
    bb_3d_vehicles = bb_3d_data[0]
    bb_3d_walkers = bb_3d_data[1]

    # Depth + bb coordinate check
    valid_bb_vehicles = proccess_3D_bb_with_depth(bb_3d_vehicles, depth_array, sensor_width, sensor_height,
                                                  camera_x_location, camera_y_location, camera_z_location)
    valid_bb_walkers = proccess_3D_bb_with_depth(bb_3d_walkers, depth_array, sensor_width, sensor_height,
                                                 camera_x_location, camera_y_location, camera_z_location)

    # Transform to np array for later easier saving
    if valid_bb_vehicles.size == 0:
        valid_bb_vehicles = np.asarray([-1, -1, -1, -1])
    if valid_bb_walkers.size == 0:
        valid_bb_walkers = np.asarray([-1, -1, -1, -1])
    valid_bbs = np.asarray([valid_bb_vehicles.ravel(), valid_bb_walkers.ravel()])  # Flattening the arrays
    return valid_bbs


def proccess_3D_bb_with_depth(bb_3d_actors, depth_array, sensor_width, sensor_height, camera_x_location,
                              camera_y_location, camera_z_location):
    valid_bbs = []
    for actor in bb_3d_actors:
        actor_bbs = []
        boxes_x_points = actor[:, 0] - camera_x_location
        boxes_y_points = actor[:, 1] - camera_y_location
        boxes_z_points = actor[:, 2] - camera_z_location
        # Check if at least one distance point is in front of the vehicle
        if any(z >= 0 for z in boxes_z_points):
            good_boxes_x_points = []
            good_boxes_y_points = []

            # Removing occluded points based on depth image
            for point in range(len(boxes_z_points)):
                x = int(boxes_x_points[point])
                y = int(boxes_y_points[point])

                # Respecting image size constraints
                if x > sensor_width:
                    x = sensor_width
                if y > sensor_height:
                    y = sensor_height
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                # Getting only points that are NO further than what the depth image shows
                if boxes_z_points[point] <= depth_array[y-1][x-1]:  # -1 to deal with python's numbering notation
                    good_boxes_x_points.append(x)
                    good_boxes_y_points.append(y)
            if (not good_boxes_y_points) and (not good_boxes_x_points):
                continue

            # Its okay to find the maximum points like this since we are dealing with CONVEX objects [boxes!]
            x_min = min(good_boxes_x_points)
            y_min = min(good_boxes_y_points)
            x_max = max(good_boxes_x_points)
            y_max = max(good_boxes_y_points)

            # Avoiding to get very small bounding boxes
            box_size = (x_max-x_min) * (y_max-y_min)
            img_size = sensor_height*sensor_width
            if (box_size / img_size) > 3E-4:  # 3E-4 was a value found by observation
                actor_bbs.append(np.array([x_min, y_min, x_max, y_max]))
        # If for that actor there are no valid bbs, then I am skipping him
        actor_bbs = np.array(actor_bbs)
        if actor_bbs.size > 0:
            valid_bbs.append(actor_bbs)
    valid_bbs = np.array(valid_bbs)
    return valid_bbs
