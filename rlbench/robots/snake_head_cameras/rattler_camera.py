from pyrep.objects.vision_sensor import VisionSensor


class RattlerCamera(object):
    def __init__(self):
        self._camera = VisionSensor('rattler_eye')
        self._camera_state = 1

    def get_camera_state(self):
        return self._camera_state

    def set_camera_properties(self) -> None:
        def _set_props(cam: VisionSensor, rgb: bool, depth: bool,
                       conf: CameraConfig):
            if not (rgb or depth):
                cam.remove()
            else:
                cam.set_resolution(conf.image_size)
                cam.set_render_mode(conf.render_mode)
        _set_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)

        if self._obs_config.left_shoulder_camera.mask:
            self._cam_shoulder_left_mask = self._cam_over_shoulder_left.copy()
            self._cam_shoulder_left_mask.set_render_mode(
                RenderMode.OPENGL_COLOR_CODED)
        if self._obs_config.right_shoulder_camera.mask:
            self._cam_shoulder_right_mask = self._cam_over_shoulder_right.copy()
            self._cam_shoulder_right_mask.set_render_mode(
                RenderMode.OPENGL_COLOR_CODED)
        if self._obs_config.wrist_camera.mask:
            self._cam_wrist_mask = self._cam_wrist.copy()
            self._cam_wrist_mask.set_render_mode(
                RenderMode.OPENGL_COLOR_CODED)