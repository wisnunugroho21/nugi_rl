import numpy as np
from eps_runner.standard import StandardRunner

from memory.list_memory import ListMemory
from environment.vectorized_env import VectorEnv

class VectorizedRunner(StandardRunner):
    def __init__(self, env, render, training_mode, n_update, agent = None, max_action = 1, writer = None, n_plot_batch = 0):
        self.env                = env
        self.agent              = agent
        self.render             = render
        self.training_mode      = training_mode
        self.n_update           = n_update
        self.max_action         = max_action
        self.writer             = writer
        self.n_plot_batch       = n_plot_batch

        self.t_updates          = 0
        self.t_aux_updates      = 0
        self.i_episode          = 0
        self.total_reward       = 0
        self.eps_time           = 0

        self.env                = VectorEnv(env)
        self.memories           = [ListMemory() for _ in range(len(env))]
        self.states             = self.env.reset()

    def run_discrete_iteration(self, agent = None, memories = None):
        if agent is None:
            agent = self.agent

        if memories is None:
            memories = self.memories

        for _ in range(self.n_update):
            actions = agent.act(self.states)
            datas   = self.env.step(action)    

            rewards     = []
            next_states = []
            for state, action, memory, data in zip(self.states, actions, memories, datas):
                next_state, reward, done, _ = data
                rewards.append(reward)
                next_states.append(next_state)
                
                if self.training_mode:
                    memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist())        
                            
            self.states         = next_states
            self.eps_time       += 1 
            total_reward        += np.mean(rewards)
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0
        
        return agent, memories

    def run_continous_iteration(self, agent = None, memories = None):
        if agent is None:
            agent = self.agent

        if memories is None:
            memories = self.memories

        for _ in range(self.n_update):
            actions = agent.act(self.states)

            action_gym = np.clip(actions, -1.0, 1.0) * self.max_action
            datas       = self.env.step(action_gym)

            rewards     = []
            next_states = []
            for state, action, memory, data in zip(self.states, actions, memories, datas):
                next_state, reward, done, _ = data
                rewards.append(reward)
                next_states.append(next_state)
                
                if self.training_mode:
                    memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist())
                
            self.states         = next_states
            self.eps_time       += 1 
            self.total_reward   += np.mean(rewards)
                    
            if self.render:
                self.env.render()

            if done:                
                self.i_episode  += 1
                print('Episode {} \t t_reward: {} \t time: {} '.format(self.i_episode, self.total_reward, self.eps_time))

                if self.i_episode % self.n_plot_batch == 0 and self.writer is not None:
                    self.writer.add_scalar('Rewards', self.total_reward, self.i_episode)
                    self.writer.add_scalar('Times', self.eps_time, self.i_episode)

                self.states         = self.env.reset()
                self.total_reward   = 0
                self.eps_time       = 0             
        
        return agent, memories