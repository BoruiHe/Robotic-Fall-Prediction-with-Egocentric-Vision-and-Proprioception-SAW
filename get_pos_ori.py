import os
import pybullet as pb
import numpy as np
from env.ergo import PoppyErgoEnv


if __name__ == "__main__":

    for folder in range(5000):
        root_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy')
        category = 'fall'
        folder = str(folder)
        
        ''' actual joint angles at each time step. Yes.'''
        ajp = np.load(os.path.join(root_path, category, folder, 'actual_joint_pos.npz'))['ajp']

        # Initiate an environment
        env = PoppyErgoEnv(pb.POSITION_CONTROL, step_hook=None, show=False)
        pos, ori = [], []

        # reset all joint angles for every time step for each Poppy joint
        timesteps = ajp.shape[0]*8 # default fps is 240, I downsampled it to 30 by saving 1 image every 8 time steps in data generation.
        for upsampled_ts in range(timesteps):

            '''calculate correct time step'''
            if upsampled_ts % 8 == 0:
                ts = upsampled_ts//8

                '''reset joint state for each joint'''
                for jointidx in range(len(env.joint_index)):
                    pb.resetJointState(env.robot_id, jointidx, ajp[ts, jointidx])

                '''immediately get link position and orientation'''
                states = pb.getLinkStates(env.robot_id, range(len(env.joint_index)), computeLinkVelocity=1)
                pos.append(np.array([state[0] for state in states])) # 0->Cartesian position
                ori.append(np.array([state[1] for state in states])) # 1->Cartesian orientation
                # print(f'{ts}: {len(pos)}')


        position = np.array(pos)
        orientation = np.array(ori)
        # print(position.shape, orientation.shape)
        # print(position)
        # print(orientation)
        np.savez_compressed(os.path.join(root_path, category, folder, 'pos.npz'), pos=position)
        np.savez_compressed(os.path.join(root_path, category, folder, 'ori.npz'), ori=orientation)
        env.close()