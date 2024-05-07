import mujoco
import mujoco.viewer
import os
import time

# 모델 로드
# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/rby_assets/rby_transfer_cube.xml')
model_path = './rby_assets/rby_transfer_cube.xml'
print(model_path)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 좌표축 표시 설정 활성화
# vopt = mujoco.MjvOption()
# vopt.frame = 1  # Frame 정보를 표시하는 옵션 활성화

# 뷰어 런칭
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    # while time.time() - start_time < 30:  # 30초 동안 유지
    while True:  # 
        mujoco.mj_step(model, data)  # 물리 시뮬레이션 스텝 진행
        
