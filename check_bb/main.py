"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 04/11/2019
Updated: 14/11/2019
"""
import cv2


def filter_bb_to_2d(bb_3d_data, depth_array, sensor_width=1024, sensor_height=768):
    # Bounding Box processing
    bb_3d_vehicles = bb_3d_data[0]
    bb_3d_walkers = bb_3d_data[1]
    print('total vehicles on world', len(bb_3d_vehicles))
    print('total walkers on world', len(bb_3d_walkers))

    # Depth + bb coordinate check
    valid_bb_vehicles = proccess_3D_bb_with_depth(bb_3d_vehicles, depth_array, sensor_width, sensor_height)
    valid_bb_walkers = proccess_3D_bb_with_depth(bb_3d_walkers, depth_array, sensor_width, sensor_height)
    valid_bb_vehicles = transform_bb_3d_to_2d(valid_bb_vehicles, sensor_width, sensor_height)
    valid_bb_walkers = transform_bb_3d_to_2d(valid_bb_walkers, sensor_width, sensor_height)

    # FIXME Saving results as imgs is optional - need to fix rgb array input reference
    save = False
    if save:
        img = False
        rgb_img = cv2.imread(rgb)
        cv2.imwrite('raw_img.jpeg', rgb_img)

        for bb in valid_bb_vehicles:
            cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
        for bb in valid_bb_walkers:
            cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        cv2.imwrite('filtered_boxed_img.jpeg', rgb_img)
        print('bb_walkers on img:', len(valid_bb_walkers))
        print('bb_vehicles on img:', len(valid_bb_vehicles))
    return valid_bb_vehicles, valid_bb_walkers


def proccess_3D_bb_with_depth(bb_3d_actors, depth_array, sensor_width, sensor_height):
    valid_bbs = []
    for actor in bb_3d_actors:
        actor_bbs = []
        for bb_xyz_point in actor:
            x = int(bb_xyz_point[0]) - 1  # -1 is accounting for the x=1 position of the camera on the vehicle
            y = int(bb_xyz_point[1])
            z = bb_xyz_point[2] - 2  # -2 is accounting for the Z=2 position of the camera on the vehicle

            # These limits stretching are done so that the vehicles' bounding boxes appear whole even if only half of
            # the car is shown
            softening_thresh = 200
            if -softening_thresh <= x < 0:
                x = 0
            elif sensor_width < x <= sensor_width + softening_thresh:
                x = sensor_width

            if -softening_thresh <= y < 0:
                y = 0
            elif sensor_height < y <= sensor_height + softening_thresh:
                y = sensor_height

            # Check if its inside the camera dimensions
            if 0 <= x <= sensor_width and 0 <= y <= sensor_height:
                # Allow for some stretching on the z axis as well (a vehicle could be driving against us and thus get
                # only half of its body)
                if -1 <= z < 0:
                    z = 0
                # Check if its behind something compared to the depth img (if its in front, i.e. nearer to us, then ok)
                if 0 <= z <= depth_array[y-1][x-1]:  # y and x on depth array are inverted for some reason
                    actor_bbs.append([x, y])
        if not actor_bbs == []:
            valid_bbs.append(actor_bbs)
    return valid_bbs


def transform_bb_3d_to_2d(bb_3d_actor, sensor_width, sensor_height):
    valid_bbs = []
    for actor in bb_3d_actor:
        x_data = [x[0] for x in actor]
        y_data = [x[1] for x in actor]
        x_min = min(x_data)
        y_min = min(y_data)
        x_max = max(x_data)
        y_max = max(y_data)
        shape = (x_max-x_min, y_max-y_min)
        size = shape[0]*shape[1]
        proportion = size/(sensor_height*sensor_width)
        # print('shape {0}; size {1}; proportion {2}'.format(shape, size, proportion))
        # Avoiding to get very small bounding boxes
        if proportion > 3E-4:  # 3E-4 was a value found by observation
            valid_bbs.append([x_min, y_min, x_max, y_max])
    return valid_bbs

