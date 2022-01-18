from gym.spaces.box import Box
import numpy as np

import glob
import os
import sys
import time
import numpy as np
import cv2
import math
import queue
from PIL import Image

try:
    sys.path.append(glob.glob('/home/nugroho/Projects/Simulator/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from nugi_rl.environment.carla.standard import CarlaEnv

class CarlaTimestepEnv(CarlaEnv):
    def __init__(self, im_height = 480, im_width = 480, im_preview = False, max_step = 512, index_pos = None):
        super().__init__(im_height, im_width, im_preview, max_step, index_pos)

    def reset(self):
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]        

        pos = self._get_pos()
        pos = carla.Transform(carla.Location(x = pos[0], y = pos[1], z = 1.0), carla.Rotation(pitch = 0, yaw = pos[2], roll = 0))
        
        self.vehicle    = self.world.spawn_actor(self.model_3, pos)        
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0, brake = 1.0, steer = 0))        
                
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, carla.Transform(carla.Location(x = 1.6, z = 1.7)), attach_to = self.vehicle)
        self.cam_sensor.listen(self.cam_queue.put)

        for _ in range(2):
            self._tick_env()
        
        self.col_sensor = self.world.spawn_actor(self.col_detector, carla.Transform(), attach_to = self.vehicle)
        self.col_sensor.listen(lambda event: self._process_collision(event))
        
        self.crl_sensor = self.world.spawn_actor(self.crl_detector, carla.Transform(), attach_to = self.vehicle)
        self.crl_sensor.listen(lambda event: self._process_crossed_line(event))
        
        self.actor_list.append(self.cam_sensor)
        self.actor_list.append(self.col_sensor)
        self.actor_list.append(self.crl_sensor)
        self.actor_list.append(self.vehicle)

        self.cur_step = 0

        images = []

        for _ in range(2):
            image_data  = np.zeros((self.im_height, self.im_width, 3), dtype = np.uint8)
            images.append(Image.fromarray(image_data, 'RGB'))

        images.append(self._process_image(self.cam_queue.get()))       

        del self.collision_hist[:]
        del self.crossed_line_hist[:] 
        
        return images, np.array([0, 0])

    def step(self, action):
        prev_loc    = self.vehicle.get_location()
        images      = []

        for _ in range(3):
            steer   = -1 if action[0] < -1 else 1 if action[0] > 1 else action[0]
            if action[1] >= 0:
                throttle    = 1 if action[1] > 1 else action[1]
                brake       = 0
            else:
                brake       = (-1 if action[1] < -1 else action[1]) * -1
                throttle    = 0

            self.vehicle.apply_control(carla.VehicleControl(steer = float(steer), throttle = float(throttle), brake = float(brake)))

            self._tick_env()
            images.append(self._process_image(self.cam_queue.get()))

        self.cur_step   += 1

        v       = self.vehicle.get_velocity()
        mps     = math.sqrt(v.x ** 2 + v.y ** 2)

        a       = self.vehicle.get_angular_velocity()
        dgs     = math.sqrt(a.x ** 2 + a.y ** 2)
                
        loc     = self.vehicle.get_location()
        dif_x   = loc.x - prev_loc.x if loc.x - prev_loc.x >= 0.03 else 0
        dif_y   = loc.y - prev_loc.y if loc.y - prev_loc.y >= 0.03 else 0
        dif_loc = math.sqrt(dif_x ** 2 + dif_y ** 2)

        done    = False
        reward  = dif_loc * 1 - 0.1       

        if self.cur_step >= self.max_step:
            done    = True

        elif len(self.crossed_line_hist) > 0 or len(self.collision_hist) > 0:
            done    = True
            reward  += -0.1 * mps

        elif loc.x >= -100 or loc.y >= -10:
            done    = True
            reward  += 100        
        
        return images, np.array([mps, dgs]), reward, done, None