"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 14/10/2019
Updated: 20/11/2019
"""

"""
Plan:
1000 frames = 8,7 GB
5000 frames = 43,5 GB
7000 frames = 60,9 GB

ok 0) Make capture between frames longer than a single frame (maybe skip 30 frames for each frame?)
ok 1) Change town (7 towns total) -> PERFORM MANUALLY, SAFER TO RECORD!
ok 2) Change weather (15 weathers total - check which ones are actually good on simulation...)
ok 3) Make a new ego vehicle spawn every n frames
TODO
ok X) Spawn npc actors only once. Put and erase only the sensor objects into different vehicles.
X) Check which weathers work best for rgb cameras. (Which ones give the most impact/show higher image quality) -> + check night weather by manually doing so
X) Check if there is a unreal setup which gives better rgb post processing img
"""

import os
import sys
import time
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver


def timer(total_time):
    for time_left in range(total_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("Sleeping for more {0} seconds".format(time_left))
        sys.stdout.flush()
        time.sleep(1)
    print('\n')
    return True


if __name__ == "__main__":
    sensor_width = 1920#1024
    sensor_height = 1080#768
    fov = 110

    # Carla settings
    print("HDF5 File opened")
    HDF5_file = HDF5Saver(sensor_width, sensor_height, os.path.join("data", "carla_dataset.hdf5"))
    CarlaWorld = CarlaWorld(HDF5_file=HDF5_file)

    timestamps = []
    egos_to_run = 3
    print('Starting to record data...')
    CarlaWorld.spawn_npcs(number_of_vehicles=150, number_of_walkers=50)
    for weather_option in CarlaWorld.weather_options:
        CarlaWorld.set_weather(weather_option)
        ego_vehicle_iteration = 0
        while ego_vehicle_iteration < egos_to_run:
            CarlaWorld.begin_data_acquisition(sensor_width, sensor_height, fov,
                                             frames_to_record_one_ego=60, timestamps=timestamps,
                                             egos_to_run=egos_to_run)
            print('Changing ego vehicle...')
            ego_vehicle_iteration += 1

#    CarlaWorld.remove_npcs()
    print('Finished simulation.')
    print('Saving timestamps...')
    CarlaWorld.HDF5_file.record_all_timestamps(timestamps)
    HDF5_file.close_HDF5()
