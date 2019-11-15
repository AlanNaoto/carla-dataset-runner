import os
import cv2
import numpy as np


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, sensorbgra.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


if __name__ == "__main__":
    path = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/dataset_collector/data"
    time = "20191106-132309"
    depth_file = os.path.join(path, "depth", "depth{0}.npy".format(time))
    data = np.load(depth_file)

    data = data.reshape((768, 1024, 4))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000

    # Saving different image types
    cv2.imwrite('data/depth_default.png', normalized_depth)

    normalized_depth = cv2.normalize(normalized_depth, normalized_depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)
    cv2.imwrite('data/depth_minmaxnorm.png', normalized_depth)

    normalized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_PINK)
    # normalized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    cv2.imwrite('data/depth_minmax_colormap.png', normalized_depth)
