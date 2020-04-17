from pyrep.const import RenderMode
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.observation_config import CameraConfig

import numpy as np

class RattlerCamera(object):
    def __init__(self):
        self._head_camera = VisionSensor('rattler_eye')
        self._head_camera_mask = None
        self._head_camera_state = 1

    def get_camera_state(self) -> int:
        return self._head_camera_state

    def set_camera_state(self, state: int) -> None:
        self._head_camera_state = state

    def set_camera_properties(self, conf: CameraConfig) -> None:
        if not (conf.rgb or conf.depth):
            self._head_camera.remove()
        else:
            self._head_camera.set_resolution(conf.image_size)
            self._head_camera.set_render_mode(conf.render_mode)

        if conf.mask:
            self._head_camera_mask = self._head_camera.copy()
            self._head_camera_mask.set_render_mode(RenderMode.OPENGL_COLOR_CODED)

    def capture_rgb(self) -> np.ndarray:
        return self._head_camera.capture_rgb()

    def capture_depth(self) -> np.ndarray:
        return self._head_camera.capture_depth()

    def mask(self) -> VisionSensor:
        return self._head_camera_mask