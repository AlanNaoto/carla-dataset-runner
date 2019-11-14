"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 14/10/2019
Updated: 06/11/2019

TODO LIST GENERAL
ok 1) Create city
ok 2) Set weather
ok 3) Spawn cars and pedestrians
ok 4) Check if transit parameters (traffic lights) work ok
ok 5) Put RGB camera, depth sensor and bbox sensors

# TODO LIST SPECIFIC
# [ok] - Fix camera position to not get car hood
# [ok] - ERASE cyclists and motorcyclists from blueprint creation, since their labeling is not adequate
# [fix it when dealing with waymo] - Define a smaller image window? -> For convenience, use same size as waymo dataset's rgb images
# [progress]- Integrate BB code into main code  -> editing HDF5Saver at the  same time
#   [progress] - Delete creation of numpy bb files (keep semantic for sanity?)
#   - Generate BB txt file
# - Generalize code to create iteratively many ambients [prob not?]
"""

"""
Weather options: {"Default", "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
 "SoftRainNoon", "ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset", "MidRainSunset", "HardRainSunset",
  "SoftRainSunset"
"""

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
    sensor_width = 1024
    sensor_height = 768
    HDF5_file = HDF5Saver(sensor_width, sensor_height)

    # Carla settings
    CarlaWorld = CarlaWorld(HDF5_file=HDF5_file)
    CarlaWorld.set_weather(choice="Default")
    CarlaWorld.spawn_npcs(number_of_vehicles=150, number_of_walkers=50)

    # Spawn EGO vehicle and begin recording data
    vehicle = CarlaWorld.spawn_vehicle()
    print('Sleeping so that vehicle doesn\'t begins recording data while floating on the air...')
    timer(2)
    fov = 110
    CarlaWorld.put_rgb_sensor(vehicle, sensor_width, sensor_height, fov)
    CarlaWorld.put_depth_sensor(vehicle, sensor_width, sensor_height, fov)
    # CarlaWorld.put_semantic_sensor(vehicle, sensor_width, sensor_height, fov)

    print('Recording data...')
    CarlaWorld.begin_data_acquisition(frames_to_record=10, sensor_width=sensor_width, sensor_height=sensor_height)
    CarlaWorld.clean_actor_list()
    HDF5_file.close_HDF5()
