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
    sys.path.append(glob.glob("/home/nugroho/Projects/Simulator/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg" % (
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
        self.front_camera       = None
        self.cur_loc            = None
        self.collision_hist     = []
        self.crossed_line_hist  = []
        self.actor_list         = []
        self.episode_start      = 0

        self.im_height              = im_height
        self.im_width               = im_width
        self.im_preview             = im_preview
        self.seconds_per_episode    = seconds_per_episode

        self.observation_space  = Box(low = -1.0, high = 1.0, shape = (im_height, im_width))
        self.action_space       = Box(low = -1.0, high = 1.0, shape = (2, 1))

        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)

        self.world          = self.client.get_world()
        blueprint_library   = self.world.get_blueprint_library()

        self.model_3        = blueprint_library.filter('model3')[0]
        self.rgb_cam        = blueprint_library.find('sensor.camera.semantic_segmentation')
        self.col_detector   = blueprint_library.find('sensor.other.collision')
        self.crl_detector   = blueprint_library.find('sensor.other.lane_invasion')

        self.rgb_cam.set_attribute('image_size_x', f'{im_height}')
        self.rgb_cam.set_attribute('image_size_y', f'{im_width}')
        
    def __del__(self):
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

    def __process_crossed_line(self, event):
        self.crossed_line_hist.append(event)

    def reset(self):
        del self.collision_hist[:]
        del self.crossed_line_hist[:]

        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]

        self.transform  = np.random.choice(self.world.get_map().get_spawn_points())
        self.vehicle    = self.world.spawn_actor(self.model_3, self.transform)
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0))
                
        self.cam_sensor = self.world.spawn_actor(self.rgb_cam, carla.Transform(carla.Location(x = 1.6, z = 1.7)), attach_to = self.vehicle)
        self.cam_sensor.listen(lambda image: self.__process_image(image))
        
        self.col_sensor = self.world.spawn_actor(self.col_detector, carla.Transform(), attach_to = self.vehicle)
        self.col_sensor.listen(lambda event: self.__process_collision(event))
        
        self.crl_sensor = self.world.spawn_actor(self.crl_detector, carla.Transform(), attach_to = self.vehicle)
        self.crl_sensor.listen(lambda event: self.__process_crossed_line(event))
        
        self.actor_list.append(self.cam_sensor)
        self.actor_list.append(self.col_sensor)
        self.actor_list.append(self.crl_sensor)
        self.actor_list.append(self.vehicle)
        
        self.cur_loc    = self.vehicle.get_location()
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        return self.front_camera

    def step(self, action):
        v0      = self.vehicle.get_velocity()
        kmh0    = 3.6 * math.sqrt(v0.x ** 2 + v0.y ** 2 + v0.z ** 2)

        if action[0] >= 0:
            if kmh0 < 0:
                self.vehicle.apply_control(carla.VehicleControl(brake = float(action[0] * 1), steer = float(action[1])))
            else:    
                self.vehicle.apply_control(carla.VehicleControl(throttle = float(action[0]), steer = float(action[1])))
        else:
            if kmh0 > 0:
                self.vehicle.apply_control(carla.VehicleControl(brake = float(action[0] * -1), steer = float(action[1])))
            else:
                self.vehicle.apply_control(carla.VehicleControl(throttle = float(action[0]), steer = float(action[1]), reverse = True))
        
        loc         = self.vehicle.get_location()
        dif         = math.sqrt((loc.x - self.cur_loc.x) ** 2 + (loc.y - self.cur_loc.y) ** 2 + (loc.z - self.cur_loc.z) ** 2)        

        done = False
        reward = 0
        if len(self.collision_hist) > 0:
            done = True
            reward = -100
        elif len(self.crossed_line_hist) > 0:
            done = True
            reward = -50
        elif dif >= 5:
            reward = 1

        if self.episode_start + self.seconds_per_episode < time.time():
            done = True

        self.loc    = loc

        return self.front_camera, reward, done, None