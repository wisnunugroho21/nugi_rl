import numpy as np
from eps_runner.standard import StandardRunner

from memory.list_memory import ListMemory
from environment.vectorized_env import VectorEnv

class VectorizedRunner(StandardRunner):
    def __init__(self, envs, render, training_mode, n_update, agent = None, max_action = 1, writer = None, n_plot_batch = 0):
        self.envs               = envs
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.states             = [env.reset() for env in self.envs]
        self.memories           = [ListMemory() for _ in range(len(envs))]

        self.states             = [env.reset() for env in envs]
        self.total_rewards      = [0 for _ in range(len(envs))]
        self.eps_times          = [0 for _ in range(len(envs))]
        self.i_episodes         = [0 for _ in range(len(envs))]

    def run_discrete_iteration(self, agent = None):
        for memory in self.memories:
            memory.clear_memory()

        for _ in range(self.n_update):
            actions = agent.act(self.states)

            for index, (env, memory, action) in enumerate(zip(self.envs, self.memories, actions)):
                next_state, reward, done, _ = env.step(int(action))

                if self.training_mode:
                    memory.save_eps(self.states[index].tolist(), action, reward, float(done), next_state.tolist())

                self.states[index]           = next_state
                self.total_rewards[index]    += reward
                self.eps_times[index]        += 1

                if self.render:
                    env.render()

                if done:
                    self.i_episodes[index]  += 1
                    print('Agent {} Episode {} \t t_reward: {} \t time: {} '.format(index, self.i_episodes[index], self.total_rewards[index], self.eps_times[index]))

                    if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                        self.writer.add_scalar('Rewards', self.total_rewards[index], self.i_episodes[index])
                        self.writer.add_scalar('Times', self.eps_times[index], self.i_episodes[index])

                    self.states[index]           = env.reset()
                    self.total_rewards[index]    = 0
                    self.eps_times[index]        = 0
        
        return self.memories

    def run_continous_iteration(self, agent = None):
        for memory in self.memories:
            memory.clear_memory()

        for _ in range(self.n_update):
            actions = agent.act(self.states)

            for index, (env, memory, action) in enumerate(zip(self.envs, self.memories, actions)):
                action_gym = np.clip(action, -1.0, 1.0) * self.max_action
                next_state, reward, done, _ = env.step(action_gym)

                if self.training_mode:
                    memory.save_eps(self.states[index].tolist(), action, reward, float(done), next_state.tolist())

                self.states[index]           = next_state
                self.total_rewards[index]    += reward
                self.eps_times[index]        += 1

                if self.render:
                    env.render()

                if done:
                    self.i_episodes[index]  += 1
                    print('Agent {} Episode {} \t t_reward: {} \t time: {} '.format(index, self.i_episodes[index], self.total_rewards[index], self.eps_times[index]))

                    if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                        self.writer.add_scalar('Rewards', self.total_rewards[index], self.i_episodes[index])
                        self.writer.add_scalar('Times', self.eps_times[index], self.i_episodes[index])

                    self.states[index]           = env.reset()
                    self.total_rewards[index]    = 0
                    self.eps_times[index]        = 0
        
        return self.memories