import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
import pyquaternion as pyq
import IPython
e = IPython.embed

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noisen
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])

class RbyPickAndTransferPolicy(BasePolicy):
    
    # def generate_trajectory(self, ts_first):
    #     init_mocap_pose_right = ts_first.observation['mocap_pose_right']
    #     init_mocap_pose_left = ts_first.observation['mocap_pose_left']

    #     box_info = np.array(ts_first.observation['env_state'])
    #     box_xyz = box_info[:3]
    #     box_quat = box_info[3:]
    #     # print(f"Generate trajectory for {box_xyz=}")

    #     # gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        
    #     # Pitch -60 deg 회전
    #     # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)
    #     # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30)
    #     gripper_pick_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30)
        
    #     # Roll 90 deg 회전
    #     # meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
    #     meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=0) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=0)
    #     meet_left_quat2 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=0) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
    #     meet_right_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
    #     meet_right_quat2 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
    #     # meet_right_quat3 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=5)
    #     # meat_right_quat2 = Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30) # * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30)

    #     # meet_xyz = box_xyz + np.array([0.1, 0.1, 0.1])
    #     meet_xyz = np.array([0, -0.5, 0.70])
        
    #     self.left_trajectory = [
    #         {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
    #         {"t": 150, "xyz": meet_xyz + np.array([0.2, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # approach meet position
    #         {"t": 350, "xyz": meet_xyz + np.array([0, 0, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
    #         {"t": 400, "xyz": meet_xyz + np.array([0, -0.035, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
    #         {"t": 460, "xyz": meet_xyz + np.array([-0.00, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
    #         {"t": 500, "xyz": meet_xyz + np.array([-0.00, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
    #         {"t": 550, "xyz": meet_xyz + np.array([-0.00, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 0}, #Close
    #         {"t": 600, "xyz": meet_xyz + np.array([0.04, -0.03, 0]), "quat": meet_left_quat2.elements, "gripper": 0}, # approach meet position
   
    #     ]
        
    #     # Pick 
    #     self.right_trajectory = [
    #         {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
    #         {"t": 150, "xyz": box_xyz + np.array([-0.065, -0.03, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach box #1
    #         {"t": 200, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach box #2
    #         {"t": 250, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Grip
    #         {"t": 270, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Grip
    #         {"t": 320, "xyz": box_xyz + np.array([-0.065, -0.03, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Lift Up
    #         # {"t": 220, "xyz": meet_xyz + np.array([-0.04, 0, -0.0]), "quat": meet_right_quat.elements, "gripper": 0}, # meet #1
    #         {"t": 440, "xyz": meet_xyz + np.array([-0.035, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 0},
    #         {"t": 460, "xyz": meet_xyz + np.array([-0.00, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 0}, # approach hand
    #         {"t": 550, "xyz": meet_xyz + np.array([-0.00, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 1}, # Open
    #         {"t": 600, "xyz": meet_xyz + np.array([-0.08, 0.0, -0.05]), "quat": meet_right_quat2.elements, "gripper": 1}, # Finish 
    #         # {"t": 400, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": gripper_pick_quat.elements, "gripper": 0}, 
    #         # {"t": 260, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": meet_right_quat.elements, "gripper": 1}, 
    #         # {"t": 400, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": meet_right_quat.elements, "gripper": 1}, 
    #     ]
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        
        # Roll 90 deg 회전
        # meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=0) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=0)
        meet_left_quat2 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=0) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
        meet_right_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
        meet_right_quat2 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
        # meet_right_quat3 = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=5)
        # meat_right_quat2 = Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30) # * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-30)

        # meet_xyz = box_xyz + np.array([0.1, 0.1, 0.1])
        meet_xyz = np.array([0, -0.5, 0.70])
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": meet_xyz + np.array([0.2, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # approach meet position
            {"t": 350, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
            {"t": 400, "xyz": meet_xyz + np.array([0.05, -0.035, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
            {"t": 460, "xyz": meet_xyz + np.array([-0.01, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
            {"t": 500, "xyz": meet_xyz + np.array([-0.01, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 1}, 
            {"t": 550, "xyz": meet_xyz + np.array([-0.01, -0.045, 0]), "quat": meet_left_quat2.elements, "gripper": 0}, #Close
            {"t": 600, "xyz": meet_xyz + np.array([0.04, -0.03, 0]), "quat": meet_left_quat2.elements, "gripper": 0}, # approach meet position
   
        ]
        
        # Pick 
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": box_xyz + np.array([-0.065, -0.03, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach box #1
            {"t": 200, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach box #2
            {"t": 250, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Grip
            {"t": 270, "xyz": box_xyz + np.array([-0.065, -0.03, -0.035]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Grip
            {"t": 320, "xyz": box_xyz + np.array([-0.065, -0.03, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 0}, # Lift Up
            # {"t": 220, "xyz": meet_xyz + np.array([-0.04, 0, -0.0]), "quat": meet_right_quat.elements, "gripper": 0}, # meet #1
            {"t": 440, "xyz": meet_xyz + np.array([-0.035, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 0},
            {"t": 460, "xyz": meet_xyz + np.array([-0.00, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 0}, # approach hand
            {"t": 550, "xyz": meet_xyz + np.array([-0.00, 0.02, -0.05]), "quat": meet_right_quat2.elements, "gripper": 1}, # Open
            {"t": 600, "xyz": meet_xyz + np.array([-0.08, 0.0, -0.05]), "quat": meet_right_quat2.elements, "gripper": 1}, # Finish 
            # {"t": 400, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": gripper_pick_quat.elements, "gripper": 0}, 
            # {"t": 260, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": meet_right_quat.elements, "gripper": 1}, 
            # {"t": 400, "xyz": meet_xyz + np.array([-0.2, 0, 0.02]), "quat": meet_right_quat.elements, "gripper": 1}, 
        ]

class RbyTestMotionPolicy(BasePolicy):
    
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 200000, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]
        
        # Pick 
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 200, "xyz": init_mocap_pose_right[:3]+np.array([0.5, 0, 0]), "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 600, "xyz": init_mocap_pose_right[:3]+np.array([0, 0.5, 0]), "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 800, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 1000, "xyz": init_mocap_pose_right[:3]+np.array([0, 0, 0.5]), "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 1200, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 1400, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:] * Quaternion(axis=[1.0, 0.0, 0.0], degrees=90), "gripper": 0},
            {"t": 1600, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 200000, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
        ]


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        
        # Pitch -60 deg 회전
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # Roll 90 deg 회전
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    elif 'sim_rby_task1_scripted' in task_name:
        env = make_ee_sim_env('sim_rby_task1_scripted')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])

            plt.ion()
        policy = None
        if 'sim_transfer_cube' in task_name:
            policy = PickAndTransferPolicy(inject_noise)
        elif 'sim_insertion' in task_name:
            policy = InsertionPolicy(inject_noise)
        elif 'sim_rby_task1_scripted' in task_name:
            policy = RbyPickAndTransferPolicy(inject_noise)
        else:
            raise NotImplementedError
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_rby_task1_scripted'
    # test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

