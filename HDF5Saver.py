import h5py
import numpy as np
import os


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, initial_dataset_size=5000):
        self.file = h5py.File(os.path.join("data", "carla_dataset.hdf5"), "w")
        # FIXME Create one dataset PER data entry. JOED/LUIS tested this method, and it was the fastest to index stuff
        self.rgb_dataset = self.file.create_dataset("rgb", (sensor_width, sensor_height, 3, initial_dataset_size),
                                                    maxshape=(sensor_width, sensor_height, 3, None), dtype="int8")
        self.depth_dataset = self.file.create_dataset("depth", (sensor_width, sensor_height, 1, initial_dataset_size),
                                                      maxshape=(sensor_width, sensor_height, 1, None), dtype="float32")
        # self.bounding_box_dataset = self.file.create_dataset("bounding_box", (100, initial_dataset_size),
        #                                                      dtype=h5py.string_dtype())
        self.timestamp = self.file.create_dataset("timestamp", (1, 5000), maxshape=(1, None), dtype="uint8")
        # Storing metadata
        self.file['sensor_width'] = sensor_width
        self.file['sensor_heigth'] = sensor_height
        self.file['simulation_synchronization_type'] = "syncd"

    def record_data(self, rgb_array, depth_array, bounding_box, frame_idx, timestamp):
        self.rgb_dataset[:, :, :, frame_idx] = rgb_array
        self.depth_dataset[:, :,  frame_idx] = depth_array
        # self.bounding_box_dataset[:, frame_idx] = bounding_box
        self.timestamp[:, frame_idx] = timestamp


    def close_HDF5(self):
        self.file.close()


def read_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        depth = file['depth']
        rgb = file['rgb']
        # print('file', file)
        # print('depth', depth[:])
        print('rgb', rgb[:, :, :, 0])
    return None


if __name__ == "__main__":
    # oi = HDF5Saver(sensor_width=1024, sensor_height=768)
    # oi.record_rgb_data(None, None)
    # oi.close_HDF5()
    afe = read_hdf5(os.path.join('data', 'carla_dataset.hdf5'))
