"""
Alan Naoto Tabata
naoto321@gmail.com
Created: 14/10/2019
Updated: 31/10/2019

ok 1) Create city
ok 2) Set weather
ok 3) Spawn cars and pedestrians
ok 4) Check if transit parameters (traffic lights) work ok
progress 5) Put RGB camera, depth sensor and instance (prob semantic) segmentation sensors
EDITED 5) Instance/Semantic -> Bounding box, since that's what WAYMO dataset provides
"""

"""
TODO
[ok] - Adapt spawn script to main
- Put BB code on vehicle main
- Generalize code to create iteratively many ambients [prob not]
- Generate text file containing respective bb coordinates and class
"""

"""
Weather options: {"Default", "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
 "SoftRainNoon", "ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset", "MidRainSunset", "HardRainSunset",
  "SoftRainSunset"
"""

import sys
import time
from multiprocessing import Process
from CarlaWorld import CarlaWorld


def timer(total_time):
    for time_left in range(total_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("sleeping for more {0} seconds".format(time_left))
        sys.stdout.flush()
        time.sleep(1)
    print('\n')
    return True


if __name__ == "__main__":
    CarlaWorld = CarlaWorld(global_sensor_tick=0.3)
    CarlaWorld.set_weather()
    CarlaWorld.spawn_npcs(number_of_vehicles=150, number_of_walkers=50)

    # Spawn EGO vehicle
    sensor_width = 1024
    sensor_height = 768
    vehicle = CarlaWorld.spawn_vehicle()
    print('Sleeping so that vehicle doesn\'t begins recording data while floating on the air.')
    timer(5)
    CarlaWorld.put_rgb_sensor(vehicle, sensor_width, sensor_height)
    CarlaWorld.put_depth_sensor(vehicle, sensor_width, sensor_height)
    # CarlaWorld.put_bb_sensor()

    print('Sleeping so that data will be captured for some time!')
    timer(10)
    CarlaWorld.clean_actor_list()

