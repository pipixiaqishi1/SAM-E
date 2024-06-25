# From https://github.com/stepjam/PyRep/blob/master/pyrep/objects/vision_sensor.py

import numpy as np

class VisionSensor():
  
  @staticmethod
  def pointcloud_from_depth_and_camera_params(
          depth: np.ndarray, extrinsics: np.ndarray,
          intrinsics: np.ndarray) -> np.ndarray:
      """Converts depth (in meters) to point cloud in word frame.
      :return: A numpy array of size (width, height, 3)
      """
      upc = _create_uniform_pixel_coords_image(depth.shape)
      pc = upc * np.expand_dims(depth, -1)
      C = np.expand_dims(extrinsics[:3, 3], 0).T
      R = extrinsics[:3, :3]
      R_inv = R.T  # inverse of rot matrix is transpose
      R_inv_C = np.matmul(R_inv, C)
      extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
      cam_proj_mat = np.matmul(intrinsics, extrinsics)
      cam_proj_mat_homo = np.concatenate(
          [cam_proj_mat, [np.array([0, 0, 0, 1])]])
      cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
      world_coords_homo = np.expand_dims(_pixel_to_world_coords(
          pc, cam_proj_mat_inv), 0)
      world_coords = world_coords_homo[..., :-1][0]
      return world_coords


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))


def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo
