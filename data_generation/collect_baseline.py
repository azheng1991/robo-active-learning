import numpy as np
import robosuite as suite
import os
import h5py
import time
from robosuite.wrappers import DataCollectionWrapper

def create_action(ee_pos_current, target_pos, gripper_action):
    pos_err = target_pos - ee_pos_current
    action = np.zeros(7)
    action[0:3] = pos_err * 15.0
    action[6] = gripper_action
    return action

def run_expert_episode(env):
    obs = env.reset()
    phase = 'APPROACH'
    
    initial_cube_pos = obs['cube_pos'].copy()
    target_drop_pos = initial_cube_pos + np.array([0.2, 0.2, 0.2])

    max_steps = 300
    grasp_step = 0
    drop_step = 0
    descend_step = 0
    for step_count in range(max_steps):
        cube_pos = obs['cube_pos']
        ee_pos = obs['robot0_eef_pos']

        gripper_action = 1.0

        if phase == 'APPROACH':
            target_pos = cube_pos + np.array([0.0, 0.0, 0.10])
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                phase = 'DESCEND'
                descend_step = step_count

        elif phase == 'DESCEND':
            target_pos = cube_pos + np.array([0.0, 0.0, -0.015])
            if np.linalg.norm(ee_pos - target_pos) < 0.02 or step_count > descend_step + 40:
                phase = 'GRASP'
                grasp_step = step_count

        elif phase == 'GRASP':
            target_pos = cube_pos + np.array([0.0, 0.0, -0.015])
            gripper_action = -1.0
            if step_count > grasp_step + 15:
                phase = 'LIFT'

        elif phase == 'LIFT':
            target_pos = initial_cube_pos + np.array([0.0, 0.0, 0.25])
            gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'MOVE'
                
        elif phase == 'MOVE':
            target_pos = target_drop_pos
            gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'DROP'
                drop_step = step_count
                
        elif phase == 'DROP':
            target_pos = target_drop_pos
            gripper_action = 1.0
            if step_count > drop_step + 10:
                phase = 'RETRACT'
                
        elif phase == 'RETRACT':
            target_pos = target_drop_pos + np.array([0.0, 0.0, 0.2])
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'DONE'

        else:
            target_pos = ee_pos
            gripper_action = 1.0

        action = create_action(ee_pos, target_pos, gripper_action)
        obs, reward, done, info = env.step(action)

        if phase == 'DONE':
            return True

    return False

def collect_data(num_episodes=100, out_dir='/tmp/robosuite_data'):
    comp_config = suite.load_composite_controller_config(robot='Panda')
    
    env = suite.make(
        env_name='Lift',
        robots='Panda',
        controller_configs=comp_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    
    env = DataCollectionWrapper(env, out_dir)

    success_count = 0
    while success_count < num_episodes:
        # randomize cube pos slightly
        success = run_expert_episode(env)
        if success:
            success_count += 1
            print(f'Collected {success_count}/{num_episodes} successful episodes')
            
    env.close()

if __name__ == '__main__':
    collect_data(100, '/mnt/c/Users/azhen/.gemini/antigravity/scratch/robo_data_optimizer/baseline_data')
