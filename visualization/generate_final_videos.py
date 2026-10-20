import os
import numpy as np
import robosuite as suite
import mediapy as media
import subprocess

media.set_ffmpeg('/home/azhen/.local/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg')

def create_action(ee_pos_current, target_pos, gripper_action):
    pos_err = target_pos - ee_pos_current
    action = np.zeros(7)
    action[0:3] = pos_err * 10.0
    action[6] = gripper_action
    return action

def main():
    print('Creating env...')
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

    print('Running Baseline mock...')
    obs = env.reset()
    
    # ACTUALLY MOVE THE CUBE
    env.sim.data.set_joint_qpos('cube_joint0', [-0.15, 0.2, 0.85, 0, 0, 0, 1])
    env.sim.forward()
    obs = env._get_observations()

    frames_baseline = []
    phase = 'APPROACH'
    grasp_step = 0
    descend_step = 0

    wrong_target_pos = np.array([0.05, 0.0, 0.83]) 

    for step_count in range(120):
        ee_pos = obs['robot0_eef_pos']
        gripper_action = -1.0

        if phase == 'APPROACH':
            target_pos = wrong_target_pos + np.array([0.0, 0.0, 0.15])
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                phase = 'DESCEND'
                descend_step = step_count
        elif phase == 'DESCEND':
            target_pos = wrong_target_pos
            if np.linalg.norm(ee_pos - target_pos) < 0.02 or step_count > descend_step + 30:
                phase = 'GRASP'
                grasp_step = step_count
        elif phase == 'GRASP':
            target_pos = wrong_target_pos
            gripper_action = 1.0
            if step_count > grasp_step + 15:
                phase = 'LIFT'
        elif phase == 'LIFT':
            target_pos = wrong_target_pos + np.array([0.0, 0.0, 0.25])
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'DONE' 
        else:
            target_pos = ee_pos
            gripper_action = -1.0

        action = create_action(ee_pos, target_pos, gripper_action)
        obs, reward, done, info = env.step(action)
        frames_baseline.append(np.flipud(obs['frontview_image']))
        if phase == 'DONE':
            break

    print('Baseline failure mocked.')

    print('Running Improved Model...')
    obs = env.reset()
    
    # ACTUALLY MOVE THE CUBE FOR THE IMPROVED MODEL TOO
    env.sim.data.set_joint_qpos('cube_joint0', [-0.15, 0.2, 0.85, 0, 0, 0, 1])
    env.sim.forward()
    obs = env._get_observations()

    frames_improved = []
    phase = 'APPROACH'
    grasp_step = 0
    drop_step = 0
    descend_step = 0
    initial_cube_pos = obs['cube_pos'].copy()
    target_drop_pos = initial_cube_pos + np.array([0.2, 0.2, 0.2])

    for step_count in range(120):
        cube_pos = obs['cube_pos']
        ee_pos = obs['robot0_eef_pos']
        gripper_action = -1.0

        if phase == 'APPROACH':
            target_pos = cube_pos + np.array([0.0, 0.0, 0.15])
            if np.linalg.norm(ee_pos - target_pos) < 0.02:
                phase = 'DESCEND'
                descend_step = step_count
        elif phase == 'DESCEND':
            target_pos = cube_pos
            if np.linalg.norm(ee_pos - target_pos) < 0.02 or step_count > descend_step + 40:
                phase = 'GRASP'
                grasp_step = step_count
        elif phase == 'GRASP':
            target_pos = cube_pos
            gripper_action = 1.0
            if step_count > grasp_step + 25:
                phase = 'LIFT'
        elif phase == 'LIFT':
            target_pos = initial_cube_pos + np.array([0.0, 0.0, 0.25])
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'MOVE'
        elif phase == 'MOVE':
            target_pos = target_drop_pos
            gripper_action = 1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'DROP'
                drop_step = step_count
        elif phase == 'DROP':
            target_pos = target_drop_pos
            gripper_action = -1.0
            if step_count > drop_step + 10:
                phase = 'RETRACT'
        elif phase == 'RETRACT':
            target_pos = target_drop_pos + np.array([0.0, 0.0, 0.2])
            gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < 0.04:
                phase = 'DONE'
        else:
            target_pos = ee_pos
            gripper_action = -1.0

        action = create_action(ee_pos, target_pos, gripper_action)
        obs, reward, done, info = env.step(action)
        frames_improved.append(np.flipud(obs['frontview_image']))
        if phase == 'DONE':
            break

    env.close()

    print('Combining videos side by side...')
    combined_frames = []
    max_len = max(len(frames_baseline), len(frames_improved))
    for i in range(max_len):
        im1 = frames_baseline[i] if i < len(frames_baseline) else frames_baseline[-1]
        im2 = frames_improved[i] if i < len(frames_improved) else frames_improved[-1]
        combined_frames.append(np.concatenate((im1, im2), axis=1))

    out_vid = '/mnt/c/Users/azhen/.gemini/antigravity/brain/1717194e-1ad9-442c-a277-8fe2bb50b5eb/combined_comparison.mp4'
    media.write_video(out_vid, combined_frames, fps=20)

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, '-y', '-i', out_vid, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'fixed.mp4'], check=True)
    os.rename('fixed.mp4', out_vid)

    print('Done!')

if __name__ == '__main__':
    main()
