import numpy as np


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


def get_4_points_max_2d_area(bb_3d_points):
    xmin = int(min(bb_3d_points[:, 0]))
    ymin = int(min(bb_3d_points[:, 1]))
    xmax = int(max(bb_3d_points[:, 0]))
    ymax = int(max(bb_3d_points[:, 1]))

    # If there is no area (e.g. its a line), then there is no bounding box!
    bbox_exists = True
    if xmin == xmax or ymin == ymax:
        bbox_exists = False
        return None, bbox_exists

    # Getting the Z point that is closer to the camera (since we are extrapolating the min/max points anyways)
    # This way, more bbox points can be salvaged after the depth filtering (since they will be considered in front of
    # the object that the depth array is seeing)
    z = min(bb_3d_points[:, 2])
    bb_3d_points = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
    return bb_3d_points, bbox_exists


def compute_bb_coords(possible_bb_3d_points):
    # bbcoords = nparray([point1, point2, point3, point4])
    # point1,2,3, or 4 = nparray([x, y, z]])
    xmin = int(min(possible_bb_3d_points[:, 0]))
    ymin = int(min(possible_bb_3d_points[:, 1]))
    xmax = int(max(possible_bb_3d_points[:, 0]))
    ymax = int(max(possible_bb_3d_points[:, 1]))
    return xmin, ymin, xmax, ymax


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
        # Its okay to use this variable for finding bbox points now because of weak referencing
        xmin, ymin, xmax, ymax = compute_bb_coords(possible_bb_3d_points)
        return xmin, ymin, xmax, ymax

    # TODO Test if/when this condition works.
    # Either get the box as it is or don't get it at all, based on how much occlusion it has
    else:
        # Compute area to see how much of it is occluded
        # possible_bb_3d_points = [x, y, z], [x,y,z], [x,y,z], [x,y,z] 
        xmin, ymin, xmax, ymax = compute_bb_coords(possible_bb_3d_points)
        common_depth = possible_bb_3d_points[0][2]
        depth_data_patch = depth_data[xmin:xmax, ymin:ymax]
        points_occlusion_status = (common_depth < depth_data_patch).sum()
        if points_occlusion_status/depth_data_patch.size > 0.50:
            return xmin, ymin, xmax, ymax
        else:
            return None, None, None, None


def get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, points_occlusion_status):
    visible_idx = points_occlusion_status.index(True)
    occluded_idxs = [x for x in range(len(points_occlusion_status)) if points_occlusion_status[x] is False]
    visible_point = possible_bb_3d_points[visible_idx]
    occluded_points = np.array([possible_bb_3d_points[x] for x in occluded_idxs])
    proposed_point_1 = np.array([visible_point[0], visible_point[1]])
    proposed_point_2 = np.array([visible_point[0], visible_point[1]])

    # Check if the new proposed points should increase or decrease in value
    x_axis_occluded_point = [x for x in occluded_points if x[0] == visible_point[0]][0]
    y_axis_occluded_point = [x for x in occluded_points if x[1] == visible_point[1]][0]
    x_tighten_bb = -1
    y_tighten_bb = -1
    if x_axis_occluded_point[0] > visible_point[0]:
        x_tighten_bb = 1
    if y_axis_occluded_point[1] > visible_point[1]:
        y_tighten_bb = 1

    # Begin tightening the points
    x_point_is_occluded = True
    y_point_is_occluded = True
    while x_point_is_occluded and y_point_is_occluded:
        x_axis_occluded_point[0] -= x_tighten_bb
        y_axis_occluded_point[1] -= y_tighten_bb
        if (0 <= occluded_point_1[2] <= depth_data[occluded_point_1[0]][occluded_point_1[1]]):
            pass
        if (0 <= occluded_point_2[2] <= depth_data[occluded_point_2[0]][occluded_point_2[1]]):
            occluded_point_still_occluded = False

    naotilt = 100
    return


