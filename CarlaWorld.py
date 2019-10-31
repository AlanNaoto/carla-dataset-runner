import sys
import os
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_DIR = os.path.join(os.path.dirname(MAIN_DIR), 'CARLA_0.9.6')
CARLA_EGG_PATH = os.path.join(CARLA_DIR, 'PythonAPI', 'carla', 'dist', 'carla-0.9.6-py3.5-linux-x86_64.egg')
sys.path.append(CARLA_EGG_PATH)
import carla
import random
import time
import numpy as np
import cv2
from spawn_npc import NPCClass
from client_bounding_boxes import ClientSideBoundingBoxes


class CarlaWorld:
    def __init__(self, global_sensor_tick):
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        print('Successfully connected to CARLA')
        # Setting synchronous mode and fixed time-step so that all sensors data captured happen at the same time
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # Time-step with 20 FPS
        # self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.global_sensor_tick = global_sensor_tick
        self.actor_list = []

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
        self.NPC.remove_npcs()
        print('Done destroying actors.')

    def spawn_npcs(self, number_of_vehicles, number_of_walkers):
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
        self.rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        # Camera calibration
        fov = 110
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.rgb_camera.calibration = calibration
        self.actor_list.append(self.rgb_camera)
        # Capture data
        # self.rgb_camera.listen(lambda img: img.save_to_disk(os.path.join('data', 'rgb', 'rgb{0}.jpeg'.format(time.strftime("%Y%m%d-%H%M%S")))))
        self.rgb_camera.listen(lambda img: self.process_rgb_img(img, sensor_width, sensor_height))

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
        self.depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(self.depth_camera)
        # self.depth_camera.listen(lambda img: img.save_to_disk(os.path.join('data', 'depth', 'depth{0}.jpeg'.format(time.strftime("%Y%m%d-%H%M%S")))), cc))
        self.depth_camera.listen(lambda data: self.save_depth_data(data))

    def save_depth_data(self, data):
        img = np.array(data.raw_data)
        np.save(os.path.join('data', 'depth', 'depth{0}'.format(time.strftime("%Y%m%d-%H%M%S"))), img)
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """

    def put_semantic_sensor(self, vehicle, sensor_width=640, sensor_height=480):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', '110')
        bp.set_attribute('sensor_tick', str(self.global_sensor_tick))
        cc = carla.ColorConverter.CityScapesPalette
        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.semantic_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(self.semantic_camera)
        self.semantic_camera.listen(lambda data: data.save_to_disk(os.path.join('data', 'semantic', 'sem{:010d}.jpeg'.format(data.frame)), cc))

    def put_bb_sensor(self):
        vehicles_on_screen = self.world.get_actors().filter('vehicle.*')
        walkers_on_screen = self.world.get_actors().filter('walker.*')
        bounding_boxes_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(vehicles_on_screen, self.rgb_camera)
        bounding_boxes_walkers = ClientSideBoundingBoxes.get_bounding_boxes(walkers_on_screen, self.rgb_camera)
        print('bounding_boxes_vehicles', bounding_boxes_vehicles)
        print('bounding_boxes_walkers', bounding_boxes_walkers)

    def process_rgb_img(self, img, sensor_width, sensor_height):
        img = np.array(img.raw_data)
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, :3]
        cv2.imwrite(os.path.join('data', 'rgb', 'rgb{0}.jpeg'.format(time.strftime("%Y%m%d-%H%M%S"))), img)
        self.put_bb_sensor()
