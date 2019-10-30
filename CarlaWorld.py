import sys
import os
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'CARLA_0.9.6')
CARLA_EGG_PATH = os.path.join(CARLA_DIR, 'PythonAPI', 'carla', 'dist', 'carla-0.9.6-py3.5-linux-x86_64.egg')
sys.path.append(CARLA_EGG_PATH)
import carla
import random
import numpy as np
import cv2
import sensor as sensorbgra
from spawn_npc import NPCClass


class CarlaWorld:
    def __init__(self, global_sensor_tick):
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        print('Successfully connected to CARLA')
        self.actor_list = []
        self.global_sensor_tick = global_sensor_tick

    def set_weather(self):
        # Changing weather https://carla.readthedocs.io/en/stable/carla_settings/
        weather_options = {"Default": carla.WeatherParameters.Default, "ClearNoon": carla.WeatherParameters.ClearNoon,
                           "CloudyNoon": carla.WeatherParameters.CloudyNoon, "WetNoon": carla.WeatherParameters.WetNoon,
                           "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
                           "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
                           "HardRainNoon": carla.WeatherParameters.HardRainNoon,
                           "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
                           "ClearSunset": carla.WeatherParameters.ClearSunset,
                           "CloudySunset": carla.WeatherParameters.CloudySunset,
                           "WetSunset": carla.WeatherParameters.WetSunset,
                           "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
                           "MidRainSunset": carla.WeatherParameters.MidRainSunset,
                           "HardRainSunset": carla.WeatherParameters.MidRainSunset,
                           "SoftRainSunset": carla.WeatherParameters.SoftRainSunset}
        weather = self.world.set_weather(weather_options['Default'])

    def clean_actor_list(self):
        print('Destroying actors...')
        for actor in self.actor_list:
            actor.destroy()
        for walker in self.npc_walker_list:
            walker.destroy()
        print('Done destroying actors.')
        self.NPC.remove_npcs()

    def spawn_npcs(self, number_of_vehicles, number_of_walkers):
        self.npc_walker_list = []
        self.NPC = NPCClass()
        self.NPC.create_npcs(number_of_vehicles, number_of_walkers)

    def spawn_vehicle(self):
        bp = self.blueprint_library.filter('model3')[0]
        current_town = self.world.get_map()
        all_spawn_points = current_town.get_spawn_points()
        failed_to_spawn = True
        while failed_to_spawn:
            try:
                print('trying to create EGO vehicle at random spawn point...')
                spawn_point = random.choice(all_spawn_points)
                vehicle = self.world.spawn_actor(bp, spawn_point)
            except:
                pass
            else:
                failed_to_spawn = False
        print('Created vehicle')
        vehicle.set_autopilot(True)
        self.actor_list.append(vehicle)
        return vehicle

    def put_rgb_sensor(self, vehicle, sensor_width=640, sensor_height=480):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', '110')
        bp.set_attribute('sensor_tick', str(self.global_sensor_tick))
        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(sensor)
        sensor.listen(lambda img: img.save_to_disk(os.path.join('data', 'rgb', 'rgb{:010d}.jpeg'.format(img.frame))))
        # sensor.listen(lambda img: process_rgb_img(img))

    def put_depth_sensor(self, vehicle, sensor_width=640, sensor_height=480):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', '110')
        bp.set_attribute('sensor_tick', str(self.global_sensor_tick))
        cc = carla.ColorConverter.Depth
        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.5))
        sensor = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(sensor)
        # sensor.listen(lambda img: img.save_to_disk(os.path.join('data', 'depth', 'depth{:010d}.jpeg'.format(img.frame)), cc))
        sensor.listen(lambda data: self.save_depth_data(data))

    def save_depth_data(self, data):
        img = np.array(data.raw_data)
        np.save(os.path.join('data', 'depth', 'depth{:010d}'.format(data.frame)), img)
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """

    def put_semantic_sensor(self, vehicle, sensor_width=640, sensor_height=480):
        """https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        Value	Tag	Converted color
        0	Unlabeled	( 0, 0, 0)
        1	Building	( 70, 70, 70)
        2	Fence	(190, 153, 153)
        3	Other	(250, 170, 160)
        4	Pedestrian	(220, 20, 60)
        5	Pole	(153, 153, 153)
        6	Road line	(157, 234, 50)
        7	Road	(128, 64, 128)
        8	Sidewalk	(244, 35, 232)
        9	Vegetation	(107, 142, 35)
        10	Car	( 0, 0, 142)
        11	Wall	(102, 102, 156)
        12	Traffic sign	(220, 220, 0)
        """
        bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', '110')
        bp.set_attribute('sensor_tick', str(self.global_sensor_tick))
        cc = carla.ColorConverter.CityScapesPalette
        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(sensor)
        sensor.listen(lambda data: data.save_to_disk(os.path.join('data', 'semantic', 'sem{:010d}.jpeg'.format(data.frame)), cc))


def process_rgb_img(img):
    """
    :param img:
    :return: show converted img from raw data to img window
    used for debugging/analyzing the frames on the go
    """
    width = 640
    height = 480
    img = np.array(img.raw_data)
    img = img.reshape((height, width, 4))
    img = img[:, :, :3]
    cv2.imshow("", img)
    cv2.waitKey(1)
    return img / 255.0
