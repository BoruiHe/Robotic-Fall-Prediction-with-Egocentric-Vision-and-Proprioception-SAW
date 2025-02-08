import sys, os
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk
import shutil
import argparse
import time
from env.ergo import PoppyErgoEnv
from env.phase_waypoints import get_waypoints, phase_waypoint_figure, render_legs


class observer():
    def __init__(self) -> None:
        self.imgs = []
        self.trajectory = []
        self.actual_joint_pos = []
        self.hc_z = [] # z coordinates of the head camera
        self.segmentation = {} # segmentation for frame-wise labels

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
    # jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]
    # for n, trajectory in enumerate(trajectories):
    #     pt.subplot(1, len(trajectories), n+1)
    #     for t, (duration, angles) in enumerate(trajectory):
    #         # if n == 0 and 0 < t < len(trajectory)-1: continue
    #         jnt_loc = env.forward_kinematics(angles)
    #         jnt_loc -= jnt_loc[env.joint_index['r_toe']]
    #         render_legs(env, jnt_loc, jnt_idx, zoffset=2*t, alpha = (t+1) / len(trajectory))
    #     pt.axis('equal')
    #     pt.axis('off')
    # if fname is not None: pt.savefig(fname)
    # pt.show()

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
    # relative positions
    states = pb.getLinkStates(env.robot_id, range(len(env.joint_index)), computeLinkVelocity=1)
    obs.pos.append(np.array([state[0] for state in states]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--number', help='the number of videos of the type', type=int, required=True)
    parser.add_argument('--base', help='the number of existing videos', type=int, required=True)
    args = parser.parse_args()
    # args = parser.parse_args('--number 1 --base 0 '.split())


    fall_dir_path = os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'fall')
    walk_dir_path = os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'walk')

    if args.base == 0:
        if os.path.exists(fall_dir_path):
            shutil.rmtree(fall_dir_path)
        os.makedirs(fall_dir_path)

        if os.path.exists(walk_dir_path):
            shutil.rmtree(walk_dir_path)
        os.makedirs(walk_dir_path)
    
    num_fall = len(os.listdir(fall_dir_path))
    num_walk = len(os.listdir(walk_dir_path))
    print(f'---Have {num_fall} fall videos and {num_walk} walk videos---')

    # pt.rcParams["text.usetex"] = False
    # pt.rcParams['font.family'] = 'serif'

    show_traj = False
    run_traj = True

    stds = [0.01, 0.015, 0.02, 0.025, 0.03,
            0.09, 0.095, 0.1, 0.105, 0.11] # 
    i = 1
    fall_count, walk_count = 0, 0
    while i <= args.number:
        obs = observer()
        std = stds[i % len(stds) - 1]
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
        
        if show_traj:   
            phase_waypoint_figure(env, waypoints)

        trajectories = get_direct_trajectories(env, waypoints)

        if show_traj:
            num_points = [10, 10, 10, 10, 10]
            draw_trajectories = constrained_interpolate(env, trajectories, num_points)
            phase_trajectory_figure(env, draw_trajectories)

        num_points = [10, 10, 2, 2, 1]
        trajectories = constrained_interpolate(env, trajectories, num_points)
        trajectories = [make_arccos_durations(traj) for traj in trajectories]

        if show_traj:
            pt.figure(figsize=(4,2))

            flatangs = np.stack([ang for traj in trajectories for (_, ang) in traj])
            offset = 0
            for traj in trajectories:
                durs, angs = zip(*traj)
                timepoints = offset + np.cumsum(durs)
                pt.plot(timepoints, np.stack(angs), 'k-')
                offset = timepoints[-1]
                pt.plot([offset]*2, [flatangs.min(), flatangs.max()], 'k:')

            pt.xlabel('Time Elapsed (s)')
            pt.ylabel('Joint Angles (rad)')
            pt.tight_layout()
            pt.savefig('traj.eps')
            pt.show()

            # show durations
            offset = 0
            for t,traj in enumerate(trajectories):
                durations, _ = zip(*traj)
                print(t, durations)
                pt.plot(np.arange(len(durations)) + offset, durations, 'ko-')
                offset += len(durations)
            pt.show()

        trajectories = extend_mirrored_trajectory(env, trajectories)
        env.close()

        if run_traj:

            env = PoppyErgoEnv(pb.POSITION_CONTROL, step_hook=hook_for_data_collection, show=False)
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

                # _, targets = trajectory[-1]
                # mad = np.fabs(targets - env.get_position()).max()
                # if mad > .005:
                #     env.goto_position(angles, duration=0.1)
                #     mad = np.fabs(angles - env.get_position()).max()
            obs.segmentation[n] = obs.segmentation[n] + (env.num_hook,)
            
            is_fall = np.array(obs.hc_z)[-1] < 0.7
            is_walk = np.array(obs.hc_z)[-1] >= 0.7
            
            if is_fall:
                if num_fall < 5000:
                    dir_path = os.path.join(fall_dir_path, str(i + args.base))
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                    os.makedirs(dir_path)

                    np.savez_compressed(os.path.join(dir_path, 'imgs.npz'), imgs=np.array(obs.imgs).astype('uint8'))
                    np.savez_compressed(os.path.join(dir_path, 'trajectory.npz'), traj=np.array(obs.trajectory))
                    np.savez_compressed(os.path.join(dir_path, 'actual_joint_pos.npz'), ajp=np.array(obs.actual_joint_pos))
                    with open(os.path.join(dir_path, 'hc_z.pkl'), 'wb') as pf:
                        pk.dump(obs.hc_z, pf)
                    with open(os.path.join(dir_path, 'seg.pkl'), 'wb') as pf:
                        pk.dump(obs.segmentation, pf)
                    i += 1
                    fall_count += 1
                    num_fall += 1
                else:
                    print(f'wasted fall video')
            elif is_walk:
                if num_walk < 5000:
                    dir_path = os.path.join(walk_dir_path, str(i + args.base))
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                    os.makedirs(dir_path)

                    np.savez_compressed(os.path.join(dir_path, 'imgs.npz'), imgs=np.array(obs.imgs).astype('uint8'))
                    np.savez_compressed(os.path.join(dir_path, 'trajectory.npz'), traj=np.array(obs.trajectory))
                    np.savez_compressed(os.path.join(dir_path, 'actual_joint_pos.npz'), ajp=np.array(obs.actual_joint_pos))
                    with open(os.path.join(dir_path, 'hc_z.pkl'), 'wb') as pf:
                        pk.dump(obs.hc_z, pf)
                    with open(os.path.join(dir_path, 'seg.pkl'), 'wb') as pf:
                        pk.dump(obs.segmentation, pf)
                    i += 1
                    walk_count += 1
                    num_walk += 1
                else:
                    print(f'wasted walk video')
            else:
                print(f'Case exception: {i+1}')

            env.close()

    print(f'---Generate {args.number} videos: {fall_count} fall videos and {walk_count} walk---')
    print(f'---Have {len(os.listdir(fall_dir_path))} fall videos and {len(os.listdir(walk_dir_path))} walk videos---')
