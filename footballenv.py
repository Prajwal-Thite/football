from gfootball.env import football_env
from gfootball.env import config
from gfootball.env import football_action_set
from gfootball.env import create_environment
import gym
import numpy as np

class FootballEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env_name="academy_pass_and_shoot_with_keeper", render_mode='rgb_array'):
        self.env = create_environment(
            env_name=env_name,
            rewards='scoring',
            render_mode=render_mode,
            render=True,                                      
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,                                      
        )                                      
        
        self.episode_steps = 0
        self.max_episode_steps = 300
        self.env.unwrapped._config['real_time'] = True
        self.env.unwrapped._config['video_quality_level'] = 2
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

def run_football_scenario():
    # Configuration setup
    cfg = config.Config({
        'level': 'tests.corner_test',
        'render': True,
        'dump_full_episodes': True,
        'tracesdir': './replays',
        'players': [
            'agent:left_players=2',
            'bot:right_players=1',
            'lazy:right_players=1'
        ]
    })

    # Create environment
    env = football_env.FootballEnv(cfg)
    actions_cnt = len(football_action_set.get_action_set(cfg))
    
    # Reset and render
    env.render()
    obs = env.reset()

    # Main game loop
    done = False
    step = 0
    while not done:
        step += 1
        actions = [(step + x) % actions_cnt for x in range(2)]
        obs, reward, done, info = env.step(actions)

    env.close()

if __name__ == "__main__":
    run_football_scenario()

