import os
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk
import random
import argparse
from env.ergo import PoppyErgoEnv
from itertools import combinations
from env.phase_waypoints import get_waypoints, render_legs


class observer():
    def __init__(self) -> None:
        self.imgs = []
        self.trajectory = []
        self.actual_joint_pos = []
        self.hc_z = [] # z coordinates of the head camera
        self.segmentation = {} # segmentation for frame-wise labels
        # self.pos = []

    def save_data(self, dir_path):
        np.savez_compressed(os.path.join(dir_path, 'imgs.npz'), imgs=np.array(obs.imgs).astype('uint8'))
        np.savez_compressed(os.path.join(dir_path, 'trajectory.npz'), traj=np.array(obs.trajectory))
        np.savez_compressed(os.path.join(dir_path, 'actual_joint_pos.npz'), ajp=np.array(obs.actual_joint_pos))
        # np.savez_compressed(os.path.join(dir_path, 'pos_real.npz'), pos=np.array(obs.pos))
        with open(os.path.join(dir_path, 'hc_z.pkl'), 'wb') as pf:
            pk.dump(obs.hc_z, pf)
        with open(os.path.join(dir_path, 'seg.pkl'), 'wb') as pf:
            pk.dump(obs.segmentation, pf)

"""
trajectory structure:
trajectories = (
    ...,
    (0, start angles), ..., (dur, mid angles), ..., (dur, final angles)
    ...)
start angles is just for reference, not part of motion.  final of one is start of next
(dur, target) is duration of motion to target
"""

# single goto_position command for each phase waypoint
#   returns trajectories = ( ..., ((0, start), (duration, final)), ...)
def get_direct_trajectories(env, waypoints):

    # move ankle before knee during swing to ensure ground clearance
    init, shift, push, kick = list(zip(*waypoints))[0]
    push_kick = push.copy()
    push_kick[env.joint_index['l_ankle_y']] = kick[env.joint_index['l_ankle_y']]

    # fast motions during swing
    trajectories = (
        ((0., init), (10, shift)),
        ((0., shift), (10, push)),
        ((0., push), (.25, push_kick)),
        ((0., push_kick), (.25, kick)),
        ((0., kick), (1, env.mirror_position(init))),
    )

    return trajectories

# direct_trajectory: ((dur, start), (dur, final))
# num_points is number of interpolated targets, including final and excluding start
def linearly_interpolate(direct_trajectory, num_points):

    _, start = direct_trajectory[0]
    total_duration, final = direct_trajectory[-1]
    duration = total_duration / num_points

    linear_trajectory = [(0, start)]
    for a, alpha in enumerate(np.linspace(1, 0, num_points+1)[1:]):
        angles = alpha * start + (1 - alpha) * final
        linear_trajectory.append((duration, angles))

    return linear_trajectory

# direct_trajectories: (..., ((dur, start), (dur, final)), ...)
# num_points[t] is number of interpolated targets for t^th trajectory, including final and excluding start
def constrained_interpolate(env, direct_trajectories, num_points):

    constrained_trajectories = []
    for t, ((_, start), (total_duration, final)) in enumerate(direct_trajectories):

        duration = total_duration / num_points[t]
    
        # translational offset from back to front toes/heels in target stance
        jnt_loc = env.forward_kinematics(final)
        toe_to_toe = jnt_loc[env.joint_index['r_toe']] - jnt_loc[env.joint_index['l_toe']]
        heel_to_heel = jnt_loc[env.joint_index['r_heel']] - jnt_loc[env.joint_index['l_heel']]
    
        constrained_trajectories.append([(0, start)])
        for _, alpha in enumerate(np.linspace(1, 0, num_points[t]+1)[1:]):
            angles = alpha * start + (1 - alpha) * final
    
            # enforce constraints
            jnt_loc = env.forward_kinematics(angles)
            if t == 0: # shift to push
                links = [env.joint_index['r_toe'], env.joint_index['r_heel']]
                targets = np.stack((
                    jnt_loc[env.joint_index['l_toe']] + toe_to_toe,
                    jnt_loc[env.joint_index['l_heel']] + heel_to_heel,
                ))
                free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)
    
            if t == 1: # shift to push
                links = [env.joint_index['l_toe']]
                targets = (jnt_loc[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)
    
            if t == 4: # kick to mirrored init
                links = [env.joint_index['l_heel']]
                targets = (jnt_loc[env.joint_index['r_heel']] - heel_to_heel)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)
    
            constrained_trajectories[t].append((duration, angles))

    return constrained_trajectories

def extend_mirrored_trajectory(env, trajectories):
    # mirror for second step
    trajectories += tuple(
        tuple(
            (duration, env.mirror_position(angles))
            for (duration, angles) in trajectory)
        for trajectory in trajectories)

    return trajectories

def phase_trajectory_figure(env, trajectories, fname=None):

    jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]

    fig = pt.figure(figsize=(6.5, 2.5), constrained_layout=True)
    gs = fig.add_gridspec(3, len(trajectories))

    (_, init) = trajectories[0][0]
    jnt_loc = env.forward_kinematics(init)
    r_toe, r_heel, l_toe, l_heel = (jnt_loc[env.joint_index[name]] for name in ('r_toe', 'r_heel', 'l_toe', 'l_heel'))
    ylo, yhi = l_heel[1] - r_toe[1], l_toe[1] - r_toe[1]
    xlo, xhi = 0, l_toe[0] - r_toe[0]
    
    for n, trajectory in enumerate(trajectories):
        fig.add_subplot(gs[:2, n])
        CoMs = np.empty((len(trajectory), 3))
        for t, (duration, angles) in enumerate(trajectory):
            CoMs[t] = env.center_of_mass(angles)
            jnt_loc = env.forward_kinematics(angles)
            CoMs[t] -= jnt_loc[env.joint_index['r_toe']]
            jnt_loc -= jnt_loc[env.joint_index['r_toe']]
            render_legs(env, jnt_loc, jnt_idx, zoffset=2*t, alpha = (t+1) / len(trajectory))
        # pt.axis('equal')
        pt.xlim([-1.1*ylo, 1.1*yhi])
        pt.ylim([-.01, .42])
        pt.title((
            "Initial $\\rightarrow$ Shift",
            "Shift $\\rightarrow$ Push",
            "Push $\\rightarrow$ Lift",
            "Lift $\\rightarrow$ Kick",
            "Kick $\\rightarrow$ Initial")[n])
        pt.axis('off')

        fig.add_subplot(gs[2, n])
        names = ('r_toe', 'r_heel', 'l_heel', 'l_toe', 'r_toe')
        if n == 1: names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
        if n == 2: names = ('r_toe', 'r_heel', 'r_toe')
        if n == 3: names = ('r_toe', 'r_heel', 'l_heel', 'r_toe')
        support_polygon = np.array([jnt_loc[env.joint_index[name]] for name in names])
        pt.plot(support_polygon[:,0], support_polygon[:,1], 'k-')
        for t in range(len(trajectory)):
            pt.plot(CoMs[t,0], CoMs[t,1], 'o', color = (1 - (t+1) / len(trajectory),)*3)
        pt.xlim([xlo - .05, xhi + .05])
        pt.ylim([-1.1*yhi, 1.1*ylo])
        pt.axis('off')
    
    if fname is not None: pt.savefig(fname)
    pt.show()

def make_arccos_durations(trajectory):
    durations, angles = zip(*trajectory)

    total_time = np.sum(durations)
    angles = np.array(angles)
    path_distance = np.cumsum(np.linalg.norm(angles[1:] - angles[:-1], axis=1))

    arccos_time = np.zeros(len(durations))
    arccos_time[1:] = np.arccos(1 - 2*path_distance / path_distance[-1]) * total_time / np.pi

    arccos_durations = np.zeros(len(durations))
    arccos_durations[1:] = arccos_time[1:] - arccos_time[:-1]

    return tuple(zip(arccos_durations, angles))

def hook_for_data_collection(env, action):
    # image capture
    rgba, _, _ = env.get_camera_image()
    obs.imgs.append(rgba)
    # actual joints position capture
    obs.actual_joint_pos.append(env.get_position())
    # trajectory capture
    obs.trajectory.append(action)
    # z-coordinates for plotting z vs. timesteps
    obs.hc_z.append(pb.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)[0][2]) # getLinkState()[0] is (x,y,z)

    # states = pb.getLinkStates(env.robot_id, range(len(env.joint_index)), computeLinkVelocity=1)
    # obs.pos.append(np.array([state[0] for state in states]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--number', help='the number of wallpaper combinations', type=int, required=True)
    # args = parser.parse_args()
    args = parser.parse_args('--number 1'.split())

    path = os.path.join(os.getcwd(), 'data', 'virtual_poppy_scenes_')
    fall_dir_path = os.path.join(path, 'fall')
    walk_dir_path = os.path.join(path, 'walk')

    try:
        os.makedirs(fall_dir_path)
        os.makedirs(walk_dir_path)
    except:
        pass
    num_fall = len(os.listdir(fall_dir_path))
    num_walk = len(os.listdir(walk_dir_path))
    print(f'---Have {num_fall} fall videos and {num_walk} walk videos---')

    texture_path = os.path.join(os.getcwd(), 'env', 'textures')
    texture_files = os.listdir(texture_path)
    texture_files.remove('sky.png')
    texture_dict = {}
    for idx in range(len(texture_files)):
        texture_dict[idx] = texture_files[idx]
    texture_combi = set(combinations(list(texture_dict.keys()), 4))
    if os.path.isfile(os.path.join(path, 'texture_combi.pkl')):
        with open(os.path.join(path, 'texture_combi.pkl'), 'rb') as pf:
            texture_combi_used = pk.load(pf)
        texture_combi -= set(texture_combi_used.keys())
    else:
        texture_combi_used = {}
    texture_combi_chosen = random.sample(list(texture_combi), args.number)

    stds = [0.01, 0.015, 0.02, 0.025, 0.03,
            0.09, 0.095, 0.1, 0.105, 0.11] # 
    i = 0
    fall_count, walk_count = 0, 0
    while i < args.number:
        wallpapers_chosen = texture_combi_chosen[i]
        texture_combi_used[wallpapers_chosen] = []
        wallpapers_chosen_ = [texture_files[idx] for idx in wallpapers_chosen] + ['sky.png']
        while fall_count < 20:
            obs = observer()
            std = stds[5:][fall_count % len(stds[5:])]
            perturbation = np.random.normal(0, std, 6)
            
            env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)
            # original
            waypoints = get_waypoints(env,
                # angle from vertical axis to flat leg in initial stance
                init_flat = .02*np.pi + perturbation[0],
                # angle for abs_y joint in initial stance
                init_abs_y = np.pi/16 + perturbation[1],
                # angle from swing leg to vertical axis in shift stance
                shift_swing = .05*np.pi + perturbation[2],
                # angle of torso towards support leg in shift stance
                shift_torso = np.pi/7 + perturbation[3],
                # angle from vertical axis to flat leg in push stance
                push_flat = -.00*np.pi + perturbation[4],#-.05*np.pi,
                # angle from swing leg to vertical axis in push stance
                push_swing = -.10*np.pi + perturbation[5],#-.01*np.pi,
            )
            # (..., (angles, oojl, error), ...)
            
            trajectories = get_direct_trajectories(env, waypoints)

            num_points = [10, 10, 2, 2, 1]
            trajectories = constrained_interpolate(env, trajectories, num_points)
            trajectories = [make_arccos_durations(traj) for traj in trajectories]

            trajectories = extend_mirrored_trajectory(env, trajectories)
            env.close()

            # if run_traj:

            env = PoppyErgoEnv(pb.POSITION_CONTROL, step_hook=hook_for_data_collection, show=False, wallpapers=wallpapers_chosen_)
            obs.segmentation[-1] = (1,)
            env.settle(waypoints[0][0], seconds=2)
            obs.segmentation[-1] = obs.segmentation[-1] + (env.num_hook - 1,)

            # for trajectory in trajectories[(1 if cycle == 0 else 0):]:
            for n, trajectory in enumerate(trajectories):
                if not (n in obs.segmentation.keys()):
                    if n == 0:
                        obs.segmentation[n] = (env.num_hook,)
                    else:
                        obs.segmentation[n] = (env.num_hook,)
                        obs.segmentation[n-1] = obs.segmentation[n-1] + (env.num_hook - 1,)
                for t, (duration, angles) in enumerate(trajectory):
                    env.goto_position(angles, duration=duration)

            obs.segmentation[n] = obs.segmentation[n] + (env.num_hook,)
            
            is_fall = np.array(obs.hc_z)[-1] < 0.7
            if is_fall:
                dir_path = os.path.join(fall_dir_path, f'f{num_fall}')
                os.makedirs(dir_path)
                obs.save_data(dir_path)
                texture_combi_used[wallpapers_chosen].append(f'f{num_fall}')
                fall_count += 1
                num_fall += 1         
            else:
                print('Unqualified fall video')

            env.close()
        
        while walk_count < 10:
            obs = observer()
            std = stds[:5][walk_count % len(stds[:5])]
            perturbation = np.random.normal(0, std, 6)
            
            env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)
            # original
            # traj_fname = 'pypot_traj1.pkl'
            waypoints = get_waypoints(env,
                # angle from vertical axis to flat leg in initial stance
                init_flat = .02*np.pi + perturbation[0],
                # angle for abs_y joint in initial stance
                init_abs_y = np.pi/16 + perturbation[1],
                # angle from swing leg to vertical axis in shift stance
                shift_swing = .05*np.pi + perturbation[2],
                # angle of torso towards support leg in shift stance
                shift_torso = np.pi/7 + perturbation[3],
                # angle from vertical axis to flat leg in push stance
                push_flat = -.00*np.pi + perturbation[4],#-.05*np.pi,
                # angle from swing leg to vertical axis in push stance
                push_swing = -.10*np.pi + perturbation[5],#-.01*np.pi,
            )
            # (..., (angles, oojl, error), ...)
            
            trajectories = get_direct_trajectories(env, waypoints)

            num_points = [10, 10, 2, 2, 1]
            trajectories = constrained_interpolate(env, trajectories, num_points)
            trajectories = [make_arccos_durations(traj) for traj in trajectories]

            trajectories = extend_mirrored_trajectory(env, trajectories)
            env.close()

            # if run_traj:

            env = PoppyErgoEnv(pb.POSITION_CONTROL, step_hook=hook_for_data_collection, show=False, wallpapers=wallpapers_chosen_)
            obs.segmentation[-1] = (1,)
            env.settle(waypoints[0][0], seconds=2)
            obs.segmentation[-1] = obs.segmentation[-1] + (env.num_hook - 1,)

            # for trajectory in trajectories[(1 if cycle == 0 else 0):]:
            for n, trajectory in enumerate(trajectories):
                if not (n in obs.segmentation.keys()):
                    if n == 0:
                        obs.segmentation[n] = (env.num_hook,)
                    else:
                        obs.segmentation[n] = (env.num_hook,)
                        obs.segmentation[n-1] = obs.segmentation[n-1] + (env.num_hook - 1,)
                for t, (duration, angles) in enumerate(trajectory):
                    env.goto_position(angles, duration=duration)

            obs.segmentation[n] = obs.segmentation[n] + (env.num_hook,)
            
            is_walk = np.array(obs.hc_z)[-1] >= 0.7
            if is_walk:
                dir_path = os.path.join(walk_dir_path, f'w{num_walk}')
                os.makedirs(dir_path)
                obs.save_data(dir_path)
                texture_combi_used[wallpapers_chosen].append(f'w{num_walk}')
                walk_count += 1
                num_walk += 1
            else:
                print('Unqualified walk video')

            env.close()
        
        i += 1

    print(f'---Generated {fall_count} fall and {walk_count} walk videos---')
    print(f'---Have {len(os.listdir(fall_dir_path))} fall videos and {len(os.listdir(walk_dir_path))} walk videos---')

    with open(os.path.join(path, 'texture_combi.pkl'), 'wb') as pf:
        pk.dump(texture_combi_used, pf)
    print(texture_combi_used)