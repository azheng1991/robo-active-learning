import numpy as np
import robosuite as suite
import mediapy as media

media.set_ffmpeg('/home/azhen/.local/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg')

def create_action(ee_pos_current, target_pos, gripper_action):
    pos_err = target_pos - ee_pos_current
    action = np.zeros(7)
    # Slow down approach slightly for more stability
    action[0:3] = pos_err * 10.0
    action[6] = gripper_action
    return action

def run_expert_episode():
    # Load the proper default composite config for Panda, which includes both OSC_POSE and GRIP
    comp_config = suite.load_composite_controller_config(robot='Panda')
    
    env = suite.make(
        env_name='Lift',
        robots='Panda',
        controller_configs=comp_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names='frontview',
        use_camera_obs=True,
        control_freq=20,
    )

    obs = env.reset()
    print('Environment initialized. Starting pick-and-place execution...')

    phase = 'APPROACH'
    frames = []
    
    initial_cube_pos = obs['cube_pos'].copy()
    target_drop_pos = initial_cube_pos + np.array([0.2, 0.2, 0.2])

    max_steps = 300
    grasp_step = 0
    drop_step = 0
    descend_step = 0
    for step_count in range(max_steps):
        cube_pos = obs['cube_pos']
        ee_pos = obs['robot0_eef_pos']

        gripper_action = -1.0

        if phase == 'APPROACH':
            # align directly above the cube
            target_pos = cube_pos + np.array([0.0, 0.0, 0.15])
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                print(f'[{step_count}] Reached APPROACH -> DESCEND')
                phase = 'DESCEND'
                descend_step = step_count

        elif phase == 'DESCEND':
            # go EXACTLY to cube center
            target_pos = cube_pos
            if np.linalg.norm(ee_pos - target_pos) < 0.02 or step_count > descend_step + 40:
                print(f'[{step_count}] Reached DESCEND -> GRASP. dist to cube: {np.linalg.norm(obs["gripper_to_cube_pos"]):.4f}')
                phase = 'GRASP'
                grasp_step = step_count

        elif phase == 'GRASP':
            target_pos = cube_pos
            gripper_action = 1.0
            # Wait longer for the gripper to firmly close
            if step_count > grasp_step + 25:
                print(f'[{step_count}] Reached GRASP -> LIFT')
                phase = 'LIFT'

        elif phase == 'LIFT':
            target_pos = initial_cube_pos + np.array([0.0, 0.0, 0.25])
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                print(f'[{step_count}] Reached LIFT -> MOVE. Cube lifted? Cube Z: {cube_pos[2]:.4f}')
                phase = 'MOVE'
                
        elif phase == 'MOVE':
            target_pos = target_drop_pos
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                print(f'[{step_count}] Reached MOVE -> DROP')
                phase = 'DROP'
                drop_step = step_count
                
        elif phase == 'DROP':
            target_pos = target_drop_pos
            gripper_action = -1.0
            if step_count > drop_step + 10:
                print(f'[{step_count}] Reached DROP -> RETRACT')
                phase = 'RETRACT'
                
        elif phase == 'RETRACT':
            target_pos = target_drop_pos + np.array([0.0, 0.0, 0.2])
            gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                print(f'[{step_count}] Reached RETRACT -> DONE')
                phase = 'DONE'

        else:
            target_pos = ee_pos
            gripper_action = -1.0

        action = create_action(ee_pos, target_pos, gripper_action)
        obs, reward, done, info = env.step(action)
        frames.append(np.flipud(obs['frontview_image']))

        if phase == 'DONE':
            print(f'Task successfully completed smoothly at step {step_count}.')
            break

    env.close()

    media.write_video('robosuite_pick_place_fixed.mp4', frames, fps=20)
    print(f'Saved successfully!')

if __name__ == '__main__':
    run_expert_episode()
