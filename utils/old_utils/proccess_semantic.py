def put_semantic_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
    # This function was in CarlaWorld.py
    # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
    bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
    bp.set_attribute('image_size_x', f'{sensor_width}')
    bp.set_attribute('image_size_y', f'{sensor_height}')
    bp.set_attribute('fov', f'{fov}')
    bp.set_attribute('sensor_tick', str(self.global_sensor_tick))
    # Adjust sensor relative position to the vehicle
    spawn_point = carla.Transform(carla.Location(x=1, z=2))
    self.semantic_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
    self.actor_list.append(self.semantic_camera)
    # self.semantic_camera.listen(lambda data: data.save_to_disk(
    #     os.path.join('data', 'semantic', 'sem{0}'.format(time.strftime("%Y%m%d-%H%M%S"))), cc))
    self.semantic_camera.listen(lambda data: self.process_semantic_img(data, sensor_width, sensor_height))


def process_semantic_img(self, img, sensor_width, sensor_height):
    # This function was in CarlaWorld.py
    #cc = carla.ColorConverter.CityScapesPalette
    #img.save_to_disk(os.path.join('data', 'semantic', 'sem{0}'.format(time.strftime("%Y%m%d-%H%M%S"))), cc)

    img = np.array(img.raw_data)
    img = img.reshape((sensor_height, sensor_width, 4))
    img = img[:, :, :3]
    np.savez(os.path.join('data', 'semantic', 'semantic{0}'.format(time.strftime("%Y%m%d-%H%M%S"))), img)


def process_semantic_data(semantic_numpy):
    # codification: https://carla.readthedocs.io/en/latest/cameras_and_sensors/#sensorcamerasemantic_segmentation
    # red_codification = {0: "Unlabeled", 4: "Pedestrian", 10: "Car"}
    semantic_numpy[(semantic_numpy != 10) & (semantic_numpy != 4)] = 0  # Background
    semantic_numpy[semantic_numpy == 4] = 127  # Pedestrian
    semantic_numpy[semantic_numpy == 10] = 255  # Car
    cv2.imwrite('semantic.png', semantic_numpy)
    return semantic_numpy

