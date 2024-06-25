from env import RenderMode, BaseEnv
import numpy as np
from gymnasium import spaces
import mujoco
from utils import MujocoUtil
import torch
import cv2

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
GRIPPER_OPEN = -0.01
GRIPPER_CLOSE = 0.01

#qpos: 18
#qvel: 18
class RbyTask2(BaseEnv):
    
    def __init__(self, args, render_mode: RenderMode):
        super().__init__(args, render_mode)

        low_bounds = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, GRIPPER_CLOSE, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, GRIPPER_CLOSE])
        high_bounds = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, GRIPPER_CLOSE, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, GRIPPER_CLOSE])
        self.action_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(self.model.nu,), dtype=np.float32)
        
        self.tounch_counter = 0
        self.min_dintance_left = 50
        self.min_dintance_right = 50
        self.sample_box_pose()
        
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.viewport = mujoco.MjrRect(0, 0, 640, 480)  # 기본 뷰포트 크기 설정
        

    def sample_box_pose(self):
        # pos="0 0 0.5"
        red_x_range = [-0.4, 0]
        blue_x_range = [0, 0.4]
        y_range = [0.1, 0.25]
        z_range = [0.55, 0.75]

        red_ranges = np.vstack([red_x_range, y_range, z_range])
        blue_ranges = np.vstack([blue_x_range, y_range, z_range])

        self.data.mocap_pos[0] = np.random.uniform(red_ranges[:, 0], red_ranges[:, 1])
        self.data.mocap_pos[1] = np.random.uniform(blue_ranges[:, 0], blue_ranges[:, 1])
        self.red_gripper = np.random.choice([GRIPPER_CLOSE, GRIPPER_OPEN])
        self.blue_gripper = np.random.choice([GRIPPER_CLOSE, GRIPPER_OPEN])
        # if self.red_gripper == GRIPPER_OPEN:
        #     print('red gripper open')
        # else:
        #     print('red gripper close')
        # if self.blue_gripper == GRIPPER_OPEN:
        #     print('blue gripper open')
        # else:
        #     print('blue gripper close')
            
        # print('sample\n'*10)
       
    
    def step(self, action):
        # 명령 업데이트 주기 확인
        self.data.ctrl[:] = action
        
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._get_reward(action)
        terminated, truncated = self._check_done()
        if action[7] == GRIPPER_CLOSE and action[15] == GRIPPER_CLOSE:
            print('done2')
            terminated = True
            truncated = True
        self.current_time += self.time_step  # Update current time
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_time = 0.0  # Reset the time
        self.sample_box_pose()
        self.tounch_counter = 0
        # observation 값을 구하고, reset_info로 빈 딕셔너리 반환
        observation = self._get_obs()
        reset_info = {}  # 필요한 경우 여기에 추가 정보 포함
        return observation, reset_info

    def initialize_view(self, render_mode, azimuth=90, elevation=-15, distance=3, lookat=np.array([0.0, 0.0, 1.0])):
        return super().initialize_view(render_mode, azimuth, elevation, distance, lookat)
    
    def _get_obs(self):
        top_image = self.render_with_camera('top')
        # cv2.imshow('top', top_image[:,:,::-1])
        left_wrist_image = self.render_with_camera('left_wrist')
        # cv2.imshow('left_wrist', left_wrist_image[:,:,::-1])
        right_wrist_image = self.render_with_camera('right_wrist')
        # cv2.imshow('right_wrist', right_wrist_image[:,:,::-1])
        data_dict={
            'qpos':self.data.qpos,
            'qvel':self.data.qvel,
            'images':{
                'top':top_image,
                'left_wrist':left_wrist_image,
                'right_wrist':right_wrist_image,
            },
        }
        return data_dict

    def render_with_camera(self, camera_name, width=640, height=480):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        self.viewport.width = width
        self.viewport.height = height

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(self.model, self.data, self.option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        
        # 이미지 추출
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width, 1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, self.viewport, self.context)
        return rgb


    def _get_reward(self, action):
        # reward = 0.01
        reward = 0
        left_tcp_position = MujocoUtil.get_site_position(self.model, self.data, 'left_tcp')
        right_tcp_position = MujocoUtil.get_site_position(self.model, self.data, 'right_tcp')
        
        red_box_xyz = self.data.mocap_pos[0]
        blue_box_xyz = self.data.mocap_pos[1]
        
        right_distance = self._calculate_distance(red_box_xyz, right_tcp_position)
        if right_distance <= self.min_dintance_left:
            self.min_dintance_right = right_distance
            reward+=0.1
            
        left_distance = self._calculate_distance(blue_box_xyz, left_tcp_position)
        if left_distance <= self.min_dintance_left:
            self.min_dintance_left = left_distance
            reward+=0.1
            
        if self.current_time > 0.5:
            if self.is_arm_contact_penalty:
                print('test')
                reward -=10
            if self.is_table_contact_penalty:
                print('test')
                reward -=10
        # print(reward)
        return reward
    
    def _calculate_distance(self, box_xyz, tcp_position, tolerance=1):
        distance = np.sqrt(np.sum(np.square(box_xyz - tcp_position)))
        return distance

    @property
    def is_table_contact_penalty(self):
        for i_contact in range(self.data.ncon):
            id_geom_1 = self.data.contact[i_contact].geom1
            id_geom_2 = self.data.contact[i_contact].geom2
            
            name_geom_1 = MujocoUtil.id_to_name(self.model, id_geom_1, 'geom')
            name_geom_2 = MujocoUtil.id_to_name(self.model, id_geom_2, 'geom')
                
            # 'left'가 'table'과 접촉하거나 'right'가 'table'과 접촉하면 감점
            if ('table' in [name_geom_1, name_geom_2]):
                if ('left' in name_geom_1 or 'left' in name_geom_2):
                    # print('l t contact')
                    self.tounch_counter+=1
                    return True
                    # print('table<->left')
                if ('right' in name_geom_1 or 'right' in name_geom_2):
                    # print('r t contact')
                    self.tounch_counter+=1
                    return True
                    # print('table<->right')
            
        return False
    
    @property
    def is_arm_contact_penalty(self):
        for i_contact in range(self.data.ncon):
            id_geom_1 = self.data.contact[i_contact].geom1
            id_geom_2 = self.data.contact[i_contact].geom2
            
            name_geom_1 = MujocoUtil.id_to_name(self.model, id_geom_1, 'geom')
            name_geom_2 = MujocoUtil.id_to_name(self.model, id_geom_2, 'geom')
                
            # 이름에 'left'가 들어가는 geom과 'right'가 들어가는 geom이 서로 접촉하면 감점
            if ('left' in name_geom_1 and 'right' in name_geom_2) or ('left' in name_geom_2 and 'right' in name_geom_1) :
                # print('left<->right')
                # print('l r contact')
                # print(name_geom_1)
                # print(name_geom_2)
                return True
        return False
    
    def _check_done(self):
        if self.min_dintance_left <0.0001 and self.min_dintance_right <0.0001:
            print(self.min_dintance_left)
            print(self.min_dintance_right)
            print('done1')
            return True, True

        # print(f'{self.data.qpos[16]=}')
        # if self.is_table_contact_penalty:
        #     print(f'{self.current_time} contact penalty')
        #     return True, False
        # print(contact_list)
        return False, False
