import glob
import math
import os
import queue
import sys

import cv2
import numpy as np
import torch
from gymnasium.spaces import Box
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor

try:
    sys.path.append(
        glob.glob(
            "/home/nugroho/Projects/Simulator/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla

from nugi_rl.environment.base import Environment


class CarlaEnv(Environment):
    def __init__(
        self,
        start_pos_rot: list,
        model_car: str = "model3",
        im_height: int = 480,
        im_width: int = 480,
        im_preview: bool = False,
        max_step: int = 512,
    ):
        self.cur_step = 0
        self.collision_hist = []
        self.crossed_line_hist = []
        self.actor_list = []

        self.start_pos_rot = start_pos_rot
        self.im_height = im_height
        self.im_width = im_width
        self.im_preview = im_preview
        self.max_step = max_step
        self.observation_space = Box(low=-1.0, high=1.0, shape=(im_height, im_width))
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, 1))

        client = carla.Client("127.0.0.1", 2000)
        self.world = client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        self.model_3 = blueprint_library.filter(model_car)[0]
        self.col_detector = blueprint_library.find("sensor.other.collision")
        self.crl_detector = blueprint_library.find("sensor.other.lane_invasion")
        self.rgb_cam = blueprint_library.find("sensor.camera.rgb")

        self.rgb_cam.set_attribute("image_size_x", f"{im_height}")
        self.rgb_cam.set_attribute("image_size_y", f"{im_width}")

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01666
        self.world.apply_settings(settings)

        self.cam_queue = queue.Queue()

    def __del__(self) -> None:
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]

    def _get_pos(self):
        idx_pos = np.random.randint(self.start_pos_rot)
        return self.start_pos_rot[idx_pos]

    def _process_image(self, image) -> Tensor:
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, -1))
        i = i[:, :, :3]

        if self.im_preview:
            cv2.imshow("", i)
            cv2.waitKey(1)

        i = Image.fromarray(i, "RGB")
        return pil_to_tensor(i)

    def _process_collision(self, event):
        self.collision_hist.append(event)

    def _process_crossed_line(self, event):
        self.crossed_line_hist.append(event)

    def _tick_env(self):
        self.world.tick()
        # time.sleep(0.01666)

    def is_discrete(self) -> bool:
        return False

    def get_obs_dim(self) -> int:
        return self.im_height * self.im_width

    def get_action_dim(self) -> int:
        return 2

    def reset(self) -> Tensor:
        for actor in self.actor_list:
            actor.destroy()
        del self.actor_list[:]

        pos = self._get_pos()
        pos = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=pos[3], yaw=pos[4], roll=pos[5]),
        )

        self.vehicle = self.world.spawn_actor(self.model_3, pos)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))

        self.cam_sensor = self.world.spawn_actor(
            self.rgb_cam,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self.vehicle,
        )
        self.cam_sensor.listen(self.cam_queue.put)

        for _ in range(8):
            self._tick_env()

        self.col_sensor = self.world.spawn_actor(
            self.col_detector, carla.Transform(), attach_to=self.vehicle
        )
        self.col_sensor.listen(lambda event: self._process_collision(event))

        self.crl_sensor = self.world.spawn_actor(
            self.crl_detector, carla.Transform(), attach_to=self.vehicle
        )
        self.crl_sensor.listen(lambda event: self._process_crossed_line(event))

        self.actor_list.append(self.cam_sensor)
        self.actor_list.append(self.col_sensor)
        self.actor_list.append(self.crl_sensor)
        self.actor_list.append(self.vehicle)

        self.cur_step = 0

        image = self._process_image(self.cam_queue.get())
        del self.collision_hist[:]
        del self.crossed_line_hist[:]

        return torch.stack([image, torch.tensor([0, 0])])

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prev_loc = self.vehicle.get_location()

        steer = -1 if action[0] < -1 else 1 if action[0] > 1 else action[0]
        if action[1] >= 0:
            throttle = 1 if action[1] > 1 else action[1]
            brake = 0
        else:
            brake = (-1 if action[1] < -1 else action[1]) * -1
            throttle = 0

        self.vehicle.apply_control(carla.VehicleControl(steer, throttle, brake))

        self._tick_env()
        self.cur_step += 1

        v = self.vehicle.get_velocity()
        kmh = math.sqrt(v.x**2 + v.y**2)

        loc = self.vehicle.get_location()
        dif_x = loc.x - prev_loc.x if loc.x - prev_loc.x >= 0.03 else 0
        dif_y = loc.y - prev_loc.y if loc.y - prev_loc.y >= 0.03 else 0
        dif_loc = math.sqrt(dif_x**2 + dif_y**2)

        done = False
        reward = dif_loc * 10 - 0.1
        image = self._process_image(self.cam_queue.get())

        if self.cur_step >= self.max_step:
            done = True

        elif len(self.crossed_line_hist) > 0 or len(self.collision_hist) > 0:
            done = True
            reward = -0.1 * kmh

        elif loc.x >= -100 or loc.y >= -10:
            done = True
            reward = 100

        return (
            torch.stack([image, torch.tensor([kmh, float(steer)])]),
            torch.tensor([reward]),
            torch.tensor([done]),
        )
