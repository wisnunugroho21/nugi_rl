#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob("D:/Projects/Simulator/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2
import math

from gym.spaces.box import Box
class CarlaEnv():
    def __init__(self, im_height = 480, im_width = 480, im_preview = False, seconds_per_episode = 1 * 60):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)

        self.world              = self.client.get_world()
        self.blueprint_library  = self.world.get_blueprint_library()
        self.model_3            = self.blueprint_library.filter('model3')[0]

        self.front_camera       = None
        self.collision_hist     = []
        self.actor_list         = []
        self.episode_start      = 0

        self.observation_space  = Box(low = -1.0, high = 1.0, shape = (im_height, im_width))
        self.action_space       = Box(low = -1.0, high = 1.0, shape = (2, 1))

        self.im_height              = im_height
        self.im_width               = im_width
        self.im_preview             = im_preview
        self.seconds_per_episode    = seconds_per_episode

    def __del__(self):
        if len(self.actor_list) > 0:
            for actor in self.actor_list:
                actor.destroy()

        self.__del__()

    def __process_image(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)

        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))
        i = i[:, :, :3]

        if self.im_preview:
            cv2.imshow('', i)
            cv2.waitKey(1)

        self.front_camera = i

    def __process_collision(self, event):
        self.collision_hist.append(event)

    def reset(self):
        if len(self.collision_hist) > 0:
            del self.collision_hist[:]

        if len(self.actor_list) > 0:
            for actor in self.actor_list:
                actor.destroy()

            del self.actor_list[:]

        self.transform  = np.random.choice(self.world.get_map().get_spawn_points())
        self.vehicle    = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        sensor_trans = carla.Transform(carla.Location(x = 2.5, z = 0.7))

        rgb_cam    = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        rgb_cam.set_attribute('image_size_x', f'{self.im_height}')
        rgb_cam.set_attribute('image_size_y', f'{self.im_width}')
        # rgb_cam.set_attribute('fov', '110')
        
        self.cam_sensor = self.world.spawn_actor(rgb_cam, sensor_trans, attach_to = self.vehicle)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda image: self.__process_image(image))

        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0))
        time.sleep(4)

        col_sensor      = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor, sensor_trans, attach_to = self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0, steer = 0))

        return self.front_camera

    def step(self, action):
        if action[0] >= 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = float(action[0]), steer = float(action[1])))
        else:
            self.vehicle.apply_control(carla.VehicleControl(brake = float(action[0] * -1), steer = float(action[1])))
        
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        if len(self.collision_hist) > 0:
            done = True
            reward = -100
        elif kmh < 50:
            done = False
            reward = 0
        else:
            done = False
            reward = 1

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True            

        return self.front_camera, reward, done, None