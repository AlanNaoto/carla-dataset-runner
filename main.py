"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 14/10/2019

Dataset creation plan
200 frames = 1,7 GB
Total planned: 20.000 frames = 170 GB
5 Towns
5 Weathers
x frames per ego vehicle
y amount of ego vehicles

Total frames planned = x * y * 5 weathers * 5 towns
20.000 / 5 towns = x * y * 5 weathers
x * y = 800
if x = 60 frames per ego, then
60 * y = 800
y ~= 13 egos

i.e., 13 egos * 60 frames * 5 weathers = 3900 frames per town
13900 * 5 towns = 19500 frames total
19500 frames ~= 165.75 GB

Suggested amount of vehicles and walkers so that traffic jam occurence is minimized
Town01 - 150 vehic 200 walk
Town02 - 50 vehic 100 walk
Town03 - 200 vehic 150 walk
Town04 - 250 vehic 100 walk
Town05 - 300 vehic 200 walk
"""

import argparse
import os
import sys
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver
from utils.create_video_on_hdf5.create_content_on_hdf5 import read_hdf5_test, treat_single_image, create_video_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdf5_file', default=None, type=str, help='name of hdf5 file to save the data')
    parser.add_argument('-wi', '--width', default=1024, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-he', '--height', default=768, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-ve', '--vehicles', default=0, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('-v', '--video', action="store_true", help="record a mp4 video on top of the recorded hdf5 file")
    parser.add_argument('-d', '--depth', action='store_true', help="show the depth video side by side with the rgb")
    args = parser.parse_args()
    assert(args.hdf5_file is not None)
    assert(args.width > 0 and args.height > 0)
    if args.vehicles == 0 and args.walkers == 0:
        print('Are you sure you don\'t want to spawn vehicles and pedestrians in the map?')

    # Sensor setup (rgb and depth share these values)
    # 1024 x 768 or 1920 x 1080 are recommended values. Higher values lead to better graphics but larger filesize
    sensor_width = args.width
    sensor_height = args.height
    fov = 90
    
    # Beginning data capture proccedure
    HDF5_file = HDF5Saver(sensor_width, sensor_height, os.path.join("data", args.hdf5_file + ".hdf5"))
    print("HDF5 File opened")
    CarlaWorld = CarlaWorld(HDF5_file=HDF5_file, town_name=None)

    timestamps = []
    egos_to_run = 13
    print('Starting to record data...')
    CarlaWorld.spawn_npcs(number_of_vehicles=args.vehicles, number_of_walkers=args.walkers)
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

    # For later visualization
    if args.video:
        create_video_sample(os.path.join('data', args.hdf5_file + ".hdf5"), show_depth=args.depth)
