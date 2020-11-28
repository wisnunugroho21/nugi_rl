from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

import numpy as np

class UnityWrapper():
    def __init__(self, env):
        self.env                = env
        self.env.reset()    

        self.behavior_name      = list(self.env.behavior_specs)[0]
        behavior_spec           = self.env.behavior_specs[self.behavior_name]        

        state_dim               = behavior_spec.observation_shapes[0][0]
        self.observation_space  = Box(low = -1.0, high = 1.0, shape = [state_dim])

        if behavior_spec.is_action_continuous():
            action_dim              = behavior_spec.action_size
            self.action_space       = Box(low = -1.0, high = 1.0, shape = [action_dim])
        
        else:
            action_dim              = behavior_spec.discrete_action_branches[0]
            self.action_space       = Discrete(action_dim)

        decision_steps, _       = self.env.get_steps(self.behavior_name)
        self.tracked_agents     = decision_steps.agent_id
    
    def reset(self):
        self.env.reset()

    def step(self):
        self.env.step()

    def get_steps(self, behavior_name):
        return self.env.get_steps(behavior_name)

    def set_actions(self, behavior_name, action):
        self.env.set_actions(behavior_name, action)

    def set_action_for_agent(self, agent_group, agent_id, action):
        self.set_action_for_agent(agent_group, agent_id, action)