import numpy as np

def filter_bounding_boxes(rgb_data, bb_data, depth_data, actor):
    good_bounding_boxes = []
    depth_data = np.transpose(depth_data)
    frame_width = 1024
    frame_height = 768

    assert actor=="vehicle" or actor=="walker"
    if actor == "vehicle":
        bounds = [-0.40*frame_width, 1.40*frame_width, -0.40*frame_height, 1.40*frame_height]
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


def adjust_points_to_img_size(width, height, bb_3d_points):
    for xyz_point in bb_3d_points:
        if xyz_point[0] < 0:
            xyz_point[0] = 0
        if xyz_point[0] > width:
            xyz_point[0] = width-1
        if xyz_point[1] < 0:
            xyz_point[1] = 0
        if xyz_point[1] > height:
            xyz_point[1] = height-1
        xyz_point[0] = xyz_point[0]
        xyz_point[1] = xyz_point[1]
    return bb_3d_points


def tighten_bbox_points(possible_bb_3d_points, depth_data):
    visible_points_status, possible_bb_3d_points = check_visible_points(possible_bb_3d_points, depth_data)
    # No points with occlusion
    if visible_points_status.count(True) == 4 or visible_points_status.count(True) == 3:
        xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data)
        color_idx = '3or4'
    
    # A pair of points occluded
    elif visible_points_status.count(True) == 2:
        xmin, ymin, xmax, ymax = get_bbox_for_2_visible_points(possible_bb_3d_points, depth_data, visible_points_status)
        color_idx = '2'

    elif visible_points_status.count(True) == 1:
        xmin, ymin, xmax, ymax = get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, visible_points_status)
        color_idx = '1'
    
    elif visible_points_status.count(True) == 0:
        xmin, ymin, xmax, ymax = None, None, None, None
        color_idx = '0'
    
    return xmin, ymin, xmax, ymax, color_idx

def check_visible_points(possible_bb_3d_points, depth_data):
    """
    visible_points=[True, True, False, False]
    """
    visible_points = []
    # Check which points are occluded
    for xyz_point in range(len(possible_bb_3d_points)):
        x = int(possible_bb_3d_points[xyz_point][0])
        y = int(possible_bb_3d_points[xyz_point][1])
        z = possible_bb_3d_points[xyz_point][2]
        depth_on_sensor = depth_data[x][y]
        point_visible = False
        if 0.0 <= z <= depth_on_sensor:
            point_visible = True
        visible_points.append(point_visible)
    return visible_points, possible_bb_3d_points


def get_4_points_max_2d_area(bb_3d_points):
    xmin = int(min(bb_3d_points[:, 0]))
    ymin = int(min(bb_3d_points[:, 1]))
    xmax = int(max(bb_3d_points[:, 0]))
    ymax = int(max(bb_3d_points[:, 1]))
    max_2d_area = (xmax - xmin) * (ymax - ymin)
    # If there is no area (e.g. its a line), then there is no bounding box!
    bbox_exists = True
    if xmin == xmax or ymin == ymax:
        bbox_exists = False
        return None, bbox_exists, 0

    # Getting the Z point that is closer to the camera (since we are extrapolating the min/max points anyways)
    # This way, more bbox points can be salvaged after the depth filtering (since they will be considered in front of
    # the object that the depth array is seeing)
    z = min(bb_3d_points[:, 2])
    bb_3d_points = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
    return bb_3d_points, bbox_exists, max_2d_area


def get_bbox_for_2_visible_points(possible_bb_3d_points, depth_data, points_occlusion_status):
    visible_idxs = [x for x in range(len(points_occlusion_status)) if points_occlusion_status[x] is True]
    occluded_idxs = [x for x in range(len(points_occlusion_status)) if points_occlusion_status[x] is False]
    visible_point_1 = possible_bb_3d_points[visible_idxs[0]]
    visible_point_2 = possible_bb_3d_points[visible_idxs[1]]

    # If both visible or occluded points are on the same axis, then bring the occluded points closer    
    if visible_point_1[0] == visible_point_2[0] or visible_point_1[1] == visible_point_2[1]:
        if visible_point_1[0] == visible_point_2[0]:
            shared_axis = 0
        elif visible_point_1[1] == visible_point_2[1]:
            shared_axis = 1
        occluded_point_1 = list(map(int, list(possible_bb_3d_points[occluded_idxs[0]])))
        occluded_point_2 = list(map(int, list(possible_bb_3d_points[occluded_idxs[1]])))

        # Check if the bbox should be increased or decreased
        tighten_bb = -1
        if occluded_point_1[shared_axis] > visible_point_1[shared_axis]:
            tighten_bb = 1
        # Begin tightening the points
        occluded_point_still_occluded = True
        while occluded_point_still_occluded:
            occluded_point_1[shared_axis] -= tighten_bb
            occluded_point_2[shared_axis] -= tighten_bb
            if (0 <= occluded_point_1[2] <= depth_data[occluded_point_1[0]][occluded_point_1[1]]) or\
            (0 <= occluded_point_2[2] <= depth_data[occluded_point_2[0]][occluded_point_2[1]]):
                occluded_point_still_occluded = False
        possible_bb_3d_points[occluded_idxs[0]] = occluded_point_1
        possible_bb_3d_points[occluded_idxs[1]] = occluded_point_2
    xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data)
    return xmin, ymin, xmax, ymax


def get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, points_occlusion_status):
    visible_idx = points_occlusion_status.index(True)
    occluded_idxs = [x for x in range(len(points_occlusion_status)) if points_occlusion_status[x] is False]
    visible_point = possible_bb_3d_points[visible_idx]
    occluded_points = np.array([possible_bb_3d_points[x] for x in occluded_idxs])

    # Check if the new proposed points should increase or decrease in value
    x_axis_occluded_point = [x for x in occluded_points if x[0] == visible_point[0]][0]
    y_axis_occluded_point = [x for x in occluded_points if x[1] == visible_point[1]][0]
    x_tighten_bb = -1
    y_tighten_bb = -1
    if x_axis_occluded_point[1] > visible_point[1]:
        x_tighten_bb = 1
    if y_axis_occluded_point[0] > visible_point[0]:
        y_tighten_bb = 1

    # Begin tightening the points
    x_point_is_occluded = True
    y_point_is_occluded = True

    while x_point_is_occluded and y_point_is_occluded:
        x_axis_occluded_point[1] -= x_tighten_bb
        y_axis_occluded_point[0] -= y_tighten_bb
        if 0 <= x_axis_occluded_point[2] <= depth_data[int(x_axis_occluded_point[0])][int(x_axis_occluded_point[1])]:
            x_point_is_occluded = False
        if 0 <= y_axis_occluded_point[2] <= depth_data[int(y_axis_occluded_point[0])][int(y_axis_occluded_point[1])]:
            y_point_is_occluded = False
    xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(
        np.array([visible_point, x_axis_occluded_point, y_axis_occluded_point]), depth_data)
    return xmin, ymin, xmax, ymax


def check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data):
    # Either get the box as it is or don't get it at all, based on how much occlusion it has
    # Compute area to see how much of it is occluded
    # possible_bb_3d_points = [x, y, z], [x,y,z], [x,y,z], [x,y,z]
    xmin, ymin, xmax, ymax = compute_bb_coords(possible_bb_3d_points)
    common_depth = possible_bb_3d_points[0][2]
    depth_data_patch = depth_data[xmin:xmax, ymin:ymax]
    visible_points_count = (common_depth < depth_data_patch).sum()
    if visible_points_count / depth_data_patch.size > 0.50:
        return xmin, ymin, xmax, ymax
    else:
        return None, None, None, None


def compute_bb_coords(possible_bb_3d_points):
    # bbcoords = nparray([point1, point2, point3, point4])
    # point1,2,3, or 4 = nparray([x, y, z]])
    xmin = int(min(possible_bb_3d_points[:, 0]))
    ymin = int(min(possible_bb_3d_points[:, 1]))
    xmax = int(max(possible_bb_3d_points[:, 0]))
    ymax = int(max(possible_bb_3d_points[:, 1]))
    return xmin, ymin, xmax, ymax
