import h5py
import numpy as np


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, file_path_to_save="data/carla_dataset.hdf5"):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self.file = h5py.File(file_path_to_save, "w")
        self.dt = h5py.special_dtype(vlen=np.dtype('float64'))
        # Creating groups to store each type of data
        self.rgb_group = self.file.create_group("rgb")
        self.depth_group = self.file.create_group("depth")
        self.bounding_box_group = self.file.create_group("bounding_box")
        self.timestamp_group = self.file.create_group("timestamps")

        # Storing metadata
        self.file.attrs['sensor_width'] = sensor_width
        self.file.attrs['sensor_height'] = sensor_height
        self.file.attrs['simulation_synchronization_type'] = "syncd"
        self.bounding_box_group.attrs['data_description'] = 'First row: vehicle. Second row: walker. Each entry in the'\
                                                            'same row are multiple actors present in the scene.'
        self.bounding_box_group.attrs['bbox_format'] = '[xmin, ymin, xmax, ymax] (top left coords; right bottom coords)' \
                                                       'the vector has been flattened; therefore the data must' \
                                                       'be captured in blocks of 4 elements'
        self.timestamp_group.attrs['time_format'] = "current time in MILISSECONDS since the unix epoch " \
                                                    "(time.time()*1000 in python3)"

    def record_data(self, rgb_array, depth_array, bounding_box, timestamp):
        timestamp = str(timestamp)
        self.rgb_group.create_dataset(timestamp, data=rgb_array)
        self.depth_group.create_dataset(timestamp, data=depth_array)
        self.bounding_box_group.create_dataset(timestamp, data=bounding_box, dtype=self.dt)

    def record_all_timestamps(self, timestamps_list):
        self.timestamp_group.create_dataset("timestamps", data=np.array(timestamps_list))

    def close_HDF5(self):
        self.file.close()
