# RBY_ACT

## Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

## Installation
```bash
# install package
pip install torchvision torch pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd detr && pip install -e .
```

## record_sim_episodes
* 시뮬레이션 환경에서 dataset 생성
* Camera는 Top, Angle, Vis 제공. joint camera view는 없음
* 여기서 카메라는 관측용, 실제 학습 데이터와 상관 x

## cv_record_sim_episodes
* 위와 동일, matplot부분만 cv로 변경
* 시뮬레이션 환경에서 dataset 생성
* Camera는 Top, Angle, Vis 제공. joint camera view는 없음
* 여기서 카메라는 관측용, 실제 학습 데이터와 상관 x

## scripted_policy
### ``Type``
* trajectory: time(1), pos(3), quat(4), gripper(1)
* mocap_pose_right, mocap_pose_left : pos(3), quat(4)
* env_state(box) : pos(3), quat(4)
* gripper: open/close

* waypoint 사이의 값은 interpolation한 값을 사용
* interpolation은 curr과 next의 비율을 통해 수행

## ee_sim_env
* End Effector Tracking 기반 시뮬레이션 환경 생성  
* top view만 생성
### ``get_observation``
* 관측결과 반환
* qpos, qvel, env_state(box), images

### ``get_reward``
* reward 반환
* Task마다 상이

### ``TransferCubeEETask``
* 오른손으로 잡아서 왼손으로 옮기는 행위 수행

#### ``Rewards``
* touch_right_gripper(박스가 오른손에 닿은 경우) = 1
* touch_right_gripper and not touch_table (박스가 오른손에 닿고, 박스가 들린 경우) = 2
* touch_left_gripper (박스가 왼손에 닿은 경우)= 3
* touch_left_gripper and not touch_table (박스가 왼손에 닿고, 박스가 들린 경우, 성공적으로 들고 있는 경우) = 4

## utils
### sample_box_pose
* 박스 위치 랜덤하게 생성