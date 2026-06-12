import os
import random
import re
import tempfile

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces
from scipy.spatial.transform import Rotation

USE_NYX = True

if USE_NYX:
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    import gs_nyx.nyx_py_renderer as npr


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OPENARM_URDF = os.path.join(ROOT, "openarm_description", "output.urdf")

LEFT_ARM_JOINTS = tuple(f"openarm_left_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
LEFT_FINGER_JOINTS = ("openarm_left_finger_joint1", "openarm_left_finger_joint2")
RIGHT_FINGER_JOINTS = ("openarm_right_finger_joint1", "openarm_right_finger_joint2")

joints_name = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_FINGER_JOINTS + RIGHT_FINGER_JOINTS
AGENT_DIM = len(joints_name)

# Genesis reports OpenArm DOFs interleaved by left/right arm joint in the URDF.
JOINT_LOWER = {
    "openarm_left_joint1": -3.4907,
    "openarm_right_joint1": -1.3963,
    "openarm_left_joint2": -3.3161,
    "openarm_right_joint2": -0.17453,
    "openarm_left_joint3": -1.5708,
    "openarm_right_joint3": -1.5708,
    "openarm_left_joint4": 0.0,
    "openarm_right_joint4": 0.0,
    "openarm_left_joint5": -1.5708,
    "openarm_right_joint5": -1.5708,
    "openarm_left_joint6": -0.7854,
    "openarm_right_joint6": -0.7854,
    "openarm_left_joint7": -0.7854,
    "openarm_right_joint7": -0.7854,
    "openarm_left_finger_joint1": -1.5708,
    "openarm_left_finger_joint2": 0.0,
    "openarm_right_finger_joint1": -1.5708,
    "openarm_right_finger_joint2": 0.0,
}
JOINT_UPPER = {
    "openarm_left_joint1": 1.3963,
    "openarm_right_joint1": 3.4907,
    "openarm_left_joint2": 0.17453,
    "openarm_right_joint2": 3.3161,
    "openarm_left_joint3": 1.5708,
    "openarm_right_joint3": 1.5708,
    "openarm_left_joint4": 2.4435,
    "openarm_right_joint4": 2.4435,
    "openarm_left_joint5": 1.5708,
    "openarm_right_joint5": 1.5708,
    "openarm_left_joint6": 0.7854,
    "openarm_right_joint6": 0.7854,
    "openarm_left_joint7": 0.7854,
    "openarm_right_joint7": 0.7854,
    "openarm_left_finger_joint1": 0.0,
    "openarm_left_finger_joint2": 1.5708,
    "openarm_right_finger_joint1": 0.0,
    "openarm_right_finger_joint2": 1.5708,
}

def _camera_look_at_transform(eye, target, up_hint):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)

    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = eye
    return transform


EEF_CAMERA_OFFSET_T = _camera_look_at_transform(
    eye=(0.20, 0.00, 0.07),
    target=(0.04, 0.00, -0.13),
    up_hint=(0.0, 1.0, 0.0),
)


class OpenArmSimplePickTask:
    def __init__(self, observation_height, observation_width, show_viewer=False):
        self.show_viewer = show_viewer
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self._build_scene(show_viewer)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)

    def _build_scene(self, show_viewer):
        if not gs._initialized:
            gs.init(backend=gs.gpu, precision="32", debug=False, logging_level="ERROR")

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.4, -1.0, 1.0),
                camera_lookat=(0.25, 0.0, 0.25),
                camera_fov=35,
                res=(self.observation_width, self.observation_height),
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane())
        self.openarm = self.scene.add_entity(
            gs.morphs.URDF(
                file=self._get_urdf_path(),
                fixed=True,
                merge_fixed_links=False,
                convexify=False,
            )
        )
        self.robot = self.openarm
        self.franka = self.openarm
        self.left_eef = self.openarm.get_link("openarm_left_ee_base_link")
        self.right_eef = self.openarm.get_link("openarm_right_ee_base_link")
        self.eef = self.right_eef

        self.cubeA = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.35, -0.10, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.7, 0.3, 0.3)),
        )
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.35, 0.10, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.3, 0.7)),
        )
        self.cubeC = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.25, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3)),
        )

        self._eef_cam_attached = False
        if USE_NYX:
            lights = [{
                "type": "directional",
                "dir": (-0.4, -0.4, -0.8),
                "color": (1.0, 1.0, 1.0),
                "intensity": 5.0,
                "shadow": True,
            }]
            self.front_cam = self.scene.add_sensor(NyxCameraOptions(
                res=(self.observation_width, self.observation_height),
                pos=(1.1, -0.7, 0.85),
                lookat=(0.25, 0.0, 0.20),
                fov=35.0,
                spp=32,
                render_mode=npr.ERenderMode.FastPathTracer,
                lights=lights,
            ))
            self.eef_cam0 = self.scene.add_sensor(NyxCameraOptions(
                res=(self.observation_width, self.observation_height),
                fov=75.0,
                near=0.02,
                far=50.0,
                entity_idx=self.openarm.idx,
                link_idx_local=self.left_eef.idx_local,
                offset_T=EEF_CAMERA_OFFSET_T,
                spp=32,
                render_mode=npr.ERenderMode.FastPathTracer,
                lights=lights,
            ))
            self.eef_cam1 = self.scene.add_sensor(NyxCameraOptions(
                res=(self.observation_width, self.observation_height),
                fov=75.0,
                near=0.02,
                far=50.0,
                entity_idx=self.openarm.idx,
                link_idx_local=self.right_eef.idx_local,
                offset_T=EEF_CAMERA_OFFSET_T,
                spp=32,
                render_mode=npr.ERenderMode.FastPathTracer,
                lights=lights,
            ))
            self.eef_cam = self.eef_cam1
            self._eef_cam_attached = True
        else:
            self.front_cam = self.scene.add_camera(
                res=(self.observation_width, self.observation_height),
                pos=(1.1, -0.7, 0.85),
                lookat=(0.25, 0.0, 0.20),
                fov=35,
                GUI=False,
            )
            self.eef_cam0 = self.scene.add_camera(
                res=(self.observation_width, self.observation_height),
                pos=(0.25, 0.15, 0.45),
                lookat=(0.25, 0.0, 0.05),
                fov=50,
                GUI=False,
            )
            self.eef_cam1 = self.scene.add_camera(
                res=(self.observation_width, self.observation_height),
                pos=(0.25, -0.15, 0.45),
                lookat=(0.25, 0.0, 0.05),
                fov=50,
                GUI=False,
            )
            self.eef_cam = self.eef_cam1

        self.scene.build()
        self.dof_indices = np.array([self.openarm.get_joint(name).dof_idx_local for name in joints_name])
        self.left_arm_dof = np.array([self.openarm.get_joint(name).dof_idx_local for name in LEFT_ARM_JOINTS])
        self.right_arm_dof = np.array([self.openarm.get_joint(name).dof_idx_local for name in RIGHT_ARM_JOINTS])
        self.left_fingers_dof = np.array([self.openarm.get_joint(name).dof_idx_local for name in LEFT_FINGER_JOINTS])
        self.right_fingers_dof = np.array([self.openarm.get_joint(name).dof_idx_local for name in RIGHT_FINGER_JOINTS])
        self.action_lower = np.array([JOINT_LOWER[name] for name in joints_name], dtype=np.float32)
        self.action_upper = np.array([JOINT_UPPER[name] for name in joints_name], dtype=np.float32)
        self.default_qpos = self._make_default_qpos()

    def _make_obs_space(self):
        return spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
            "observation.images.front": spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            ),
            "observation.images.eef_cam0": spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            ),
            "observation.images.eef_cam1": spaces.Box(
                low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8
            ),
        })

    def _get_urdf_path(self):
        if not USE_NYX:
            return OPENARM_URDF

        description_root = os.path.dirname(OPENARM_URDF)
        with open(OPENARM_URDF, "r", encoding="utf-8") as f:
            urdf = f.read()

        urdf = urdf.replace("package://openarm_description/", f"{description_root}/")
        urdf = urdf.replace(
            "assets/robot/openarm_v2.0/meshes/body/visual/body_link0.dae",
            "assets/robot/openarm_v2.0/meshes/body/visual/body_link0.stl",
        )
        urdf = re.sub(
            r"assets/robot/openarm_v2\.0/meshes/arm/visual/([^\"/]+)\.dae",
            r"assets/robot/openarm_v2.0/meshes/arm/collision/\1.stl",
            urdf,
        )
        urdf = re.sub(
            r"assets/end_effector/pinch_gripper/meshes/visual/([^\"/]+)\.dae",
            r"assets/end_effector/pinch_gripper/meshes/collision/\1.stl",
            urdf,
        )

        path = os.path.join(tempfile.gettempdir(), "openarm_output_nyx.urdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(urdf)
        return path

    def _make_default_qpos(self):
        qpos = np.zeros(self.openarm.n_dofs, dtype=np.float32)
        for name in joints_name:
            idx = self.openarm.get_joint(name).dof_idx_local
            lower = JOINT_LOWER[name]
            upper = JOINT_UPPER[name]
            qpos[idx] = np.clip(0.0, lower, upper)
        qpos[self.openarm.get_joint("openarm_left_joint4").dof_idx_local] = 0.8
        qpos[self.openarm.get_joint("openarm_right_joint4").dof_idx_local] = 0.8
        return qpos

    def _scale_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.action_lower + 0.5 * (action + 1.0) * (self.action_upper - self.action_lower)

    def set_random_state(self, target, x_range, y_range, z):
        x = self._random.uniform(x_range[0], x_range[1])
        y = self._random.uniform(y_range[0], y_range[1])
        pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        target.set_pos(pos_tensor)
        target.set_quat(quat_tensor)

    def reset(self):
        self.color = random.choice(["red", "blue", "green"])
        self.set_random_state(self.cubeA, (0.15, 0.45), (-0.22, 0.05), 0.04)
        self.set_random_state(self.cubeB, (0.15, 0.45), (-0.05, 0.22), 0.04)
        self.set_random_state(self.cubeC, (0.15, 0.45), (-0.18, 0.18), 0.04)

        kp = np.full(self.openarm.n_dofs, 800.0, dtype=np.float32)
        kv = np.full(self.openarm.n_dofs, 80.0, dtype=np.float32)
        kp[self.left_fingers_dof] = 50.0
        kp[self.right_fingers_dof] = 50.0
        kv[self.left_fingers_dof] = 5.0
        kv[self.right_fingers_dof] = 5.0
        self.openarm.set_dofs_kp(kp)
        self.openarm.set_dofs_kv(kv)
        self.openarm.set_dofs_force_range(
            np.array([-40, -40, -40, -40, -27, -27, -27, -27, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7]),
            np.array([40, 40, 40, 40, 27, 27, 27, 27, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
        )
        qpos_tensor = torch.tensor(self.default_qpos, dtype=torch.float32, device=gs.device)
        self.openarm.set_qpos(qpos_tensor, zero_velocity=True)
        self.openarm.control_dofs_position(qpos_tensor[self.dof_indices], self.dof_indices)
        if not self._eef_cam_attached:
            self._set_eef_cam_pos()

        self.scene.step()
        self._start_camera_recording(self.front_cam)
        self._start_camera_recording(self.eef_cam0)
        self._start_camera_recording(self.eef_cam1)
        return self.get_obs(), {}

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        target_qpos = self._scale_action(action)
        target_tensor = torch.tensor(target_qpos, dtype=torch.float32, device=gs.device)
        self.openarm.control_dofs_position(target_tensor, self.dof_indices)
        if not self._eef_cam_attached:
            self._set_eef_cam_pos()
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        terminated = True if reward == 1.0 else False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        if self.color == "red":
            pos = self.cubeA.get_pos().cpu().numpy()
        elif self.color == "blue":
            pos = self.cubeB.get_pos().cpu().numpy()
        elif self.color == "green":
            pos = self.cubeC.get_pos().cpu().numpy()
        else:
            raise ValueError(f"Invalid color: {self.color}. Choose from 'red', 'blue', or 'green'.")

        left_distance = np.linalg.norm(self.left_eef.get_pos().cpu().numpy() - pos)
        right_distance = np.linalg.norm(self.right_eef.get_pos().cpu().numpy() - pos)
        distance = min(left_distance, right_distance)
        reward = 0.5 * np.exp(-5 * distance)
        height = pos[2] - 0.025
        reward += 0.5 * (1 - np.exp(-height))
        return reward

    def get_obs(self):
        left_pos = self.left_eef.get_pos().cpu().numpy()
        left_rot = self.left_eef.get_quat().cpu().numpy()
        right_pos = self.right_eef.get_pos().cpu().numpy()
        right_rot = self.right_eef.get_quat().cpu().numpy()
        dof_pos = self.openarm.get_dofs_position().cpu().numpy()
        fingers = dof_pos[np.concatenate([self.left_fingers_dof, self.right_fingers_dof])]
        agent_pos = np.concatenate([left_pos, left_rot, right_pos, right_rot, fingers]).astype(np.float32)

        front_pixels = self._read_camera_rgb(self.front_cam)
        assert front_pixels.ndim == 3, f"front_pixels shape {front_pixels.shape} is not 3D (H, W, 3)"
        eef_pixels0 = self._read_camera_rgb(self.eef_cam0)
        assert eef_pixels0.ndim == 3, f"eef_pixels0 shape {eef_pixels0.shape} is not 3D (H, W, 3)"
        eef_pixels1 = self._read_camera_rgb(self.eef_cam1)
        assert eef_pixels1.ndim == 3, f"eef_pixels1 shape {eef_pixels1.shape} is not 3D (H, W, 3)"
        return {
            "agent_pos": agent_pos,
            "observation.images.front": front_pixels,
            "observation.images.eef_cam0": eef_pixels0,
            "observation.images.eef_cam1": eef_pixels1,
        }

    def save_videos(self, file_name, fps=30):
        if USE_NYX:
            return
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.eef_cam0.stop_recording(save_to_filename=f"{file_name}_eef_cam0.mp4", fps=fps)
        self.eef_cam1.stop_recording(save_to_filename=f"{file_name}_eef_cam1.mp4", fps=fps)

    def close(self):
        gs.destroy()

    def _set_eef_cam_pos(self):
        self._set_link_cam_pos(self.eef_cam0, self.left_eef)
        self._set_link_cam_pos(self.eef_cam1, self.right_eef)

    def _set_link_cam_pos(self, camera, link):
        eef_pos = link.get_pos().cpu().numpy()
        eef_rot = link.get_quat().cpu().numpy()
        eef_rot = eef_rot[[1, 2, 3, 0]]
        eef_transform = np.eye(4)
        eef_transform[:3, :3] = Rotation.from_quat(eef_rot).as_matrix()
        eef_transform[:3, 3] = eef_pos
        camera.set_pose(transform=eef_transform @ EEF_CAMERA_OFFSET_T)

    def _read_camera_rgb(self, camera):
        if USE_NYX and hasattr(camera, "read"):
            rgb = camera.read().rgb
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb
        return camera.render()[0]

    def _start_camera_recording(self, camera):
        if USE_NYX:
            return
        if hasattr(camera, "start_recording"):
            camera.start_recording()


if __name__ == "__main__":
    import cv2
    import time
    gs.init(backend=gs.gpu, precision="32")
    task = OpenArmSimplePickTask(observation_height=512, observation_width=512, show_viewer=False)
    task.reset()
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        obs, _, _, _, _ = task.step(action)
        cv2.imwrite("openarm_front_image.png", obs["observation.images.front"])
        cv2.imwrite("openarm_eef_cam0_image.png", obs["observation.images.eef_cam0"])
        cv2.imwrite("openarm_eef_cam1_image.png", obs["observation.images.eef_cam1"])
        time.sleep(1.0)
