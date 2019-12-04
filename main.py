"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 14/10/2019

Plan
200 frames = 1,7 GB
Total: 20.000 frames = 170 GB
5 Towns
5 Weathers
x frames per ego vehicle
y amount of ego vehicles

Frames per town = x * y * 5
20.000 / 5 towns = x * y * 5
x * y = 800
if x = 60 frames per ego, then
60 * y = 800
y = 13 egos

i.e., 13 egos * 60 frames * 5 weathers = 3900 frames per town
13900 * 5 towns = 19500 frames total

19500 frames ~= 165.75 GB

Town01 - 150 vehic 200 walk
Town02 - 100 vehic 100 walk
Town03 - 200 vehic 150 walk
Town04 - 250 vehic 100 walk
Town05 - 300 vehic 200 walk


"""

import os
import sys
import time
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver
from utils.create_video_on_hdf5.create_content_on_hdf5 import read_hdf5_test, treat_single_image, create_video_sample



def timer(total_time):
    for time_left in range(total_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("Sleeping for more {0} seconds".format(time_left))
        sys.stdout.flush()
        time.sleep(1)
    print('\n')
    return True


if __name__ == "__main__":
# 1024 x 768 or 1920 x 1080 recommended
    sensor_width = 1024
    sensor_height = 768
    fov = 90

    # Carla settings
    hdf5_file = "carla_dataset_town02.hdf5"
    print("HDF5 File opened")
    HDF5_file = HDF5Saver(sensor_width, sensor_height, os.path.join("data", hdf5_file))
    CarlaWorld = CarlaWorld(HDF5_file=HDF5_file, town_name=None)

    timestamps = []
    egos_to_run = 13
    print('Starting to record data...')
    CarlaWorld.spawn_npcs(number_of_vehicles=100, number_of_walkers=200)
    for weather_option in CarlaWorld.weather_options:
        CarlaWorld.set_weather(weather_option)
        ego_vehicle_iteration = 0
        while ego_vehicle_iteration < egos_to_run:
            CarlaWorld.begin_data_acquisition(sensor_width, sensor_height, fov,
                                             frames_to_record_one_ego=60, timestamps=timestamps,
                                             egos_to_run=egos_to_run)
            print('Setting another vehicle as EGO.')
            ego_vehicle_iteration += 1

    CarlaWorld.remove_npcs()
    print('Finished simulation.')
    print('Saving timestamps...')
    CarlaWorld.HDF5_file.record_all_timestamps(timestamps)
    HDF5_file.close_HDF5()

    # Only for visualization purposes
    create_video_sample(os.path.join('data', hdf5_file), show_depth=False)
