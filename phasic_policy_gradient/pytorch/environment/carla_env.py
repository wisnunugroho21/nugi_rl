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
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
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

IM_HEIGHT = 640
IM_WIDTH = 480

def process_image(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow('', i3)
    cv2.waitKey(1)

    return i3 / 255.0

actor_list = []

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 0.0))

    actor_list.append(vehicle)

    cam_bp  = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', f'{IM_HEIGHT}')
    cam_bp.set_attribute('image_size_y', f'{IM_WIDTH}')
    cam_bp.set_attribute('fov', '110')

    spawn_point     = carla.Transform(carla.Location(x = 2.5, z = 0.7))

    sensor          = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_image(data))

    time.sleep(5)

finally:
    for actor in actor_list:
        actor.destroy()
    print('All cleaned Up!')

class CarlaEnv():
    im_height   = IM_HEIGHT
    im_width    = IM_WIDTH
    im_preview  = IM_PREVIEW

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.world              = self.client.get_world()
        self.blueprint_library  = self.world.get_blueprint_library()
        self.model_3            = self.blueprint_library.filter('model3')[0]

        self.front_camera       = None
        self.collision_hist     = []
        self.actor_list         = []
        self.episode_start      = 0

    def __process_image(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i = i[:, :, :3]

        if self.im_preview:
            cv2.imshow('', i)
            cv2.waitKey(1)

        self.front_camera = i

    def __process_collision(self, event):
        self.collision_hist.append(event)

    def reset(self):
        self.collision_hist = []
        self.actor_list     = []

        self.transform  = np.random.choice(self.world.get_map().get_spawn_points())
        self.vehicle    = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        sensor_trans = carla.Transform(carla.Location(x = 2.5, z = 0.7))

        rgb_cam    = self.blueprint_library.find('sensor.camera.rgb')
        rgb_cam.set_attribute('image_size_x', f'{IM_HEIGHT}')
        rgb_cam.set_attribute('image_size_y', f'{IM_WIDTH}')
        rgb_cam.set_attribute('fov', '110')
        
        self.cam_sensor = self.world.spawn_actor(rgb_cam, sensor_trans, attach_to = vehicle)
        self.actor_list.append(self.cam_sensor)
        self.cam_sensor.listen(lambda image: self.__process_image(image))

        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0))
        time.sleep(4)

        col_sensor      = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor, sensor_trans, attach_to = vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, steer = 0.0))

        return self.front_camera

    def step(self, action):
        if action[0] >= 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = action[0], steer = action[1]))
        else:
            self.vehicle.apply_control(carla.VehicleControl(brake = action[0] * -1, steer = action[1]))
        
        rewards = 0
        done = False
        return self.front_camera, rewards, done, None

