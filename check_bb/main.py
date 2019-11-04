"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 04/11/2019
Updated: 04/11/2019
"""
import numpy as np
import cv2


rgb = "test_imgs/rgb20191104-171700.jpeg"
data_file = "test_imgs/bb20191104-171700.npz"
sensor_width = 1024
sensor_height = 768


def process_bb(bb_data_numpy):
    valid_bb_vehicles = []

    for bb_3d in bb_data_numpy:
        bb_2d = [x[:-1] for x in bb_3d]  # Excluding Z axis information
        vehicle_bb = []
        for bb in bb_2d:
            bb = [int(x) for x in bb]
            # Checking if there are bboxes out of the image view
            if 0 <= bb[0] <= sensor_width and 0 <= bb[1] <= sensor_height:
                vehicle_bb.append(bb[0])
                vehicle_bb.append(bb[1])

        if vehicle_bb:
            even_values = vehicle_bb[::2]
            odd_values = vehicle_bb[1::2]
            x_min = min(even_values)
            y_min = min(odd_values)
            x_max = max(even_values)
            y_max = max(odd_values)
            valid_bb_vehicles.append([x_min, y_min, x_max, y_max])
    return valid_bb_vehicles


if __name__ == "__main__":
    # Bounding Box processing
    data = np.load(data_file)
    bb_vehicles = data['arr_0']
    bb_walkers = data['arr_1']
    valid_bb_vehicles = process_bb(bb_vehicles)
    valid_bb_walkers = process_bb(bb_walkers)

    # RGB processing
    rgb_img = cv2.imread(rgb)
    for bb in valid_bb_vehicles:
        cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
    for bb in valid_bb_walkers:
        cv2.rectangle(rgb_img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

    # cv2.imshow('naoto', rgb_img)
    # cv2.waitKey(0)
    cv2.imwrite('boxed_img.png', rgb_img)


    # TODO next step: compare image data with segmentation output to check whether the object exists in the camera plane
    # TODO or not (filter data out of range)
