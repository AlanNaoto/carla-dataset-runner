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
from set_synchronous_mode import CarlaSyncMode
from bb_filter import apply_filters_to_3d_bb


class CarlaWorld:
    def __init__(self, HDF5_file):
        self.HDF5_file = HDF5_file
        # Carla initialization
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        self.world = client.get_world()
        print('Successfully connected to CARLA')
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

    def set_weather(self, choice="Default"):
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
        self.world.set_weather(weather_options[choice])

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
        bp = self.blueprint_library.filter('audi')[0]
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

    def put_rgb_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', '110')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1, z=2))
        self.rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.rgb_camera.calibration = calibration  # Parameter K of the camera
        self.actor_list.append(self.rgb_camera)
        # Capture data
        # self.rgb_camera.listen(lambda img: img.save_to_disk(os.path.join('data', 'rgb', 'rgb{0}.jpeg'.format(time.strftime("%Y%m%d-%H%M%S")))))
        # self.rgb_camera.listen(lambda img: self.process_rgb_img(img, sensor_width, sensor_height))

    def put_depth_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1, z=2))
        self.depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.actor_list.append(self.depth_camera)
        # cc = carla.ColorConverter.Depth
        # self.depth_camera.listen(lambda img: img.save_to_disk(os.path.join('data', 'depth', 'depth{0}.jpeg'.format(time.strftime("%Y%m%d-%H%M%S")))), cc))
        # self.depth_camera.listen(lambda data: self.save_depth_data(data))

    def process_depth_data(self, data):
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """
        data = np.array(data.raw_data)
        data = data.reshape((768, 1024, 4))
        data = data.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized_depth * 1000
        return depth_meters

    def get_bb_data(self):
        vehicles_on_world = self.world.get_actors().filter('vehicle.*')
        walkers_on_world = self.world.get_actors().filter('walker.*')
        bounding_boxes_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(vehicles_on_world, self.rgb_camera)
        bounding_boxes_walkers = ClientSideBoundingBoxes.get_bounding_boxes(walkers_on_world, self.rgb_camera)
        return [bounding_boxes_vehicles, bounding_boxes_walkers]

    def process_rgb_img(self, img, sensor_width, sensor_height):
        img = np.array(img.raw_data)
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, :3]
        bb = self.get_bb_data()
        return img, bb

    def begin_data_acquisition(self, frames_to_record, sensor_width, sensor_height):
        recorded_frames = 0
        timestamps = []
        with CarlaSyncMode(self.world, self.rgb_camera, self.depth_camera, fps=30) as sync_mode:
            while True:
                if recorded_frames == frames_to_record:
                    print('\n')
                    print('Saving timestamps...')
                    self.HDF5_file.record_all_timestamps(timestamps)
                    return
                # Advance the simulation and wait for the data.
                data = sync_mode.tick(timeout=2.0)  # If needed, self.frame can be obtained too
                _, rgb_data, depth_data = data
                recorded_frames += 1

                # Processing raw data
                rgb_array, bounding_box = self.process_rgb_img(rgb_data, sensor_width, sensor_height)
                depth_array = self.process_depth_data(depth_data)
                bounding_box = apply_filters_to_3d_bb(bounding_box, depth_array, sensor_width, sensor_height)
                timestamp = round(time.time()*1000.0)
                timestamps.append(timestamp)

                # Saving into opened HDF5 dataset file
                self.HDF5_file.record_data(rgb_array, depth_array, bounding_box, timestamp)
                sys.stdout.write("\r")
                sys.stdout.write('Frame {0}/{1}'.format(recorded_frames, frames_to_record))
                sys.stdout.flush()
