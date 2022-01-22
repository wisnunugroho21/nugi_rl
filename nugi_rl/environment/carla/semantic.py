import numpy as np

from nugi_rl.environment.carla.standard import CarlaEnv

class CarlaSemanticEnv(CarlaEnv):
    def __init__(self, start_pos_rot: list, model_car: str = 'model3', im_height: int = 480, im_width: int = 480, im_preview: bool = False, max_step: int = 512):
        super().__init__(start_pos_rot, model_car, im_height, im_width, im_preview, max_step)

        blueprint_library   = self.world.get_blueprint_library()
        self.rgb_cam        = blueprint_library.find('sensor.camera.semantic_segmentation')

        self.rgb_cam.set_attribute('image_size_x', f'{im_height}')
        self.rgb_cam.set_attribute('image_size_y', f'{im_width}')

    def _process_image(self, image):
        i = np.array(image.raw_data)        
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, 0]

        return i