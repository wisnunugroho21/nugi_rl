import retro

import gym
import numpy as np
import pybullet_envs

def main():
    env = gym.make("HumanoidBulletEnv-v0")
    env.render(mode="human")

    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()