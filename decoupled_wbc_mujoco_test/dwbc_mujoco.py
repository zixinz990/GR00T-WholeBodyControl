"""Headless, ROS-free MuJoCo port of the GR00T Decoupled WBC stack (decoupled_wbc).

Physics replicates decoupled_wbc/control/envs/g1/sim/base_sim.py (DefaultEnv.sim_step) together
with the Unitree SDK bridge: scene_43dof.xml, 200 Hz simulation, PD torques computed inside the
simulator from the last low-level command (kp/kd from MOTOR_KP/MOTOR_KD, dq target 0, tau 0),
clipping to motor_effort_limit_list, ctrl = torques (FREE_BASE False).

The controller replicates the control loop run_g1_control_loop.py in --sim_sync_mode:
    G1Env.observe -> G1DecoupledWholeBodyPolicy (InterpolationPolicy upper body +
    G1GearWbcPolicy lower body) -> G1Env.queue_action (JointSafetyMonitor startup ramp,
    joint-order -> actuator-order mapping for body and Dex3 hands).
G1GearWbcPolicy is ported to numpy because .venv_sim has no torch; validate_policy_parity.py
checks the port against the repo class.
"""

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

REPO = Path(__file__).resolve().parent.parent
DW = REPO / "decoupled_wbc"
SCENE = DW / "control/robot_model/model_data/g1/scene_43dof.xml"
WBC_YAML = DW / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml"
POLICY_DIR = DW / "sim2mujoco/resources/robots/g1"
# configs.py: wbc_model_path default "policy/...-Balance.onnx,policy/...-Walk.onnx"
MODEL_PATHS = ("policy/GR00T-WholeBodyControl-Balance.onnx", "policy/GR00T-WholeBodyControl-Walk.onnx")
SIM_FREQUENCY = 200      # configs.py BaseConfig.sim_frequency -> SIMULATE_DT = 1 / 200
CONTROL_FREQUENCY = 50   # configs.py BaseConfig.control_frequency

# Pinocchio joint order of g1_29dof_with_hand.urdf (fixed base, nq = 43), i.e. RobotModel's
# configuration order. Checked with pinocchio.buildModelFromUrdf.
PIN_JOINTS = (
    [f"left_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"right_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                    "wrist_roll", "wrist_pitch", "wrist_yaw")]
    + [f"left_hand_{j}_joint" for j in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")]
    + [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                     "wrist_roll", "wrist_pitch", "wrist_yaw")]
    + [f"right_hand_{j}_joint" for j in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")]
)
PIN_INDEX = {n: i for i, n in enumerate(PIN_JOINTS)}
# g1_supplemental_info.py: body_actuated_joints (= motor order, JOINT2MOTOR is the identity)
BODY_ACTUATED = [n for n in PIN_JOINTS if "_hand_" not in n]
# g1_supplemental_info.py: left/right_hand_actuated_joints (Dex3 SDK motor order)
HAND_ACTUATED = {s: [f"{s}_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
                 for s in ("left", "right")}
# joint groups (sorted indices, RobotModel.get_joint_group_indices), waist_location = lower_body
GROUP = {
    "body": sorted(PIN_INDEX[n] for n in BODY_ACTUATED),
    "lower_body": sorted(PIN_INDEX[n] for n in BODY_ACTUATED[:15]),
    "upper_body": sorted(PIN_INDEX[n] for n in PIN_JOINTS[15:]),
    "arms_and_hands": sorted(PIN_INDEX[n] for n in PIN_JOINTS[15:]),  # JointSafetyMonitor velocity_limits keys
}
# command_sender.py HandCommandSender gains (motor order)
HAND_KP = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
HAND_KD = np.array([0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])


def load_wbc_config():
    with open(WBC_YAML) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["SIMULATE_DT"] = 1.0 / SIM_FREQUENCY  # override_wbc_config
    return cfg


def load_policy_config(wbc_cfg):
    # gear_wbc_utils.load_config
    with open(REPO / wbc_cfg["GEAR_WBC_CONFIG"]) as f:
        cfg = yaml.safe_load(f)
    for key in ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]:
        cfg[key] = np.array(cfg[key], dtype=np.float32)
    return cfg


class DwbcSim:
    """DefaultEnv (base_sim.py) + UnitreeSdk2Bridge, without DDS/ROS. Elastic band not used."""

    def __init__(self, wbc_cfg):
        self.cfg = wbc_cfg
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.opt.timestep = wbc_cfg["SIMULATE_DT"]
        self.d = mujoco.MjData(self.m)
        names = [self.m.joint(i).name for i in range(self.m.njnt)]
        self.body_joint_index = np.array([i for i, n in enumerate(names) if any(
            p in n for p in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"])])
        self.left_hand_index = np.array([i for i, n in enumerate(names) if "left_hand" in n])
        self.right_hand_index = np.array([i for i, n in enumerate(names) if "right_hand" in n])
        assert [names[i] for i in self.body_joint_index] == BODY_ACTUATED
        self.mj_hand_joint_names = {"left": [names[i] for i in self.left_hand_index],
                                    "right": [names[i] for i in self.right_hand_index]}
        self.kp = np.array(wbc_cfg["MOTOR_KP"], dtype=float)
        self.kd = np.array(wbc_cfg["MOTOR_KD"], dtype=float)
        self.torque_limit = np.array(wbc_cfg["motor_effort_limit_list"])
        self.torques = np.zeros(self.m.nu)
        self.body_cmd_q = None   # no low_cmd received yet -> kp = kd = 0 -> zero torque
        self.hand_cmd_q = {"left": None, "right": None}
        self.published = None

    def reset(self, qpos):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.published = self.prepare_obs()

    def prepare_obs(self):
        d = self.d
        obs = {
            "floating_base_pose": d.qpos[:7].copy(),
            "floating_base_vel": d.qvel[:6].copy(),
            "body_q": d.qpos[self.body_joint_index + 7 - 1].copy(),
            "body_dq": d.qvel[self.body_joint_index + 6 - 1].copy(),
            "time": d.time,
        }
        for s, idx in (("left", self.left_hand_index), ("right", self.right_hand_index)):
            obs[f"{s}_hand_q"] = d.qpos[idx + 7 - 1].copy()
            obs[f"{s}_hand_dq"] = d.qvel[idx + 6 - 1].copy()
        return obs

    def queue_command(self, body_q, left_hand_q, right_hand_q):
        self.body_cmd_q = np.asarray(body_q, dtype=float).copy()
        self.hand_cmd_q = {"left": np.asarray(left_hand_q, dtype=float).copy(),
                           "right": np.asarray(right_hand_q, dtype=float).copy()}

    def sim_step(self):
        d = self.d
        self.published = self.prepare_obs()  # PublishLowState happens before the physics step
        body_tau = np.zeros(len(self.body_joint_index))
        if self.body_cmd_q is not None:
            q = d.qpos[self.body_joint_index + 7 - 1]
            dq = d.qvel[self.body_joint_index + 6 - 1]
            body_tau = self.kp * (self.body_cmd_q - q) + self.kd * (0.0 - dq)
        self.torques[self.body_joint_index - 1] = body_tau
        for s, idx in (("left", self.left_hand_index), ("right", self.right_hand_index)):
            tau = np.zeros(len(idx))
            if self.hand_cmd_q[s] is not None:
                # bridge: hand motor i drives the i-th MuJoCo joint of that hand
                tau = HAND_KP * (self.hand_cmd_q[s] - d.qpos[idx + 7 - 1]) + HAND_KD * (0.0 - d.qvel[idx + 6 - 1])
            self.torques[idx - 1] = tau
        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)
        d.ctrl[:] = self.torques
        mujoco.mj_step(self.m, d)


class GearWbcPolicyNp:
    """numpy port of decoupled_wbc/control/policy/g1_gear_wbc_policy.py (G1GearWbcPolicy)."""

    def __init__(self, cfg):
        self.config = cfg
        sess = [ort.InferenceSession(str(POLICY_DIR / p)) for p in MODEL_PATHS]
        self.policy_1, self.policy_2 = sess
        self.obs_history = deque(maxlen=cfg["obs_history_len"])
        self.obs_buffer = np.zeros(cfg["num_obs"], dtype=np.float32)
        self.use_policy_action = True  # equivalent to pressing "]" before t = 0
        self.action = np.zeros(cfg["num_actions"], dtype=np.float32)
        self.cmd = cfg["cmd_init"].copy()
        self.height_cmd = cfg["height_cmd"]
        self.roll_cmd, self.pitch_cmd, self.yaw_cmd = cfg["rpy_cmd"]
        self.observation = None

    @staticmethod
    def gravity_orientation(quat):
        # gear_wbc_utils.get_gravity_orientation / quat_rotate_inverse
        w, x, y, z = quat
        qc = np.array([w, -x, -y, -z])
        v = np.array([0.0, 0.0, -1.0])
        return np.array([
            v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
            + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3]) + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
            v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
            + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2) + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
            v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
            + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1]) + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
        ])

    def compute_observation(self, observation):
        cfg = self.config
        body_indices = GROUP["body"]
        n_joints = len(body_indices)
        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()
        quat = observation["floating_base_pose"][3:7].copy()
        omega = observation["floating_base_vel"][3:6].copy()
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[: len(cfg["default_angles"])] = cfg["default_angles"]
        single_obs = np.zeros(86, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * cfg["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega * cfg["ang_vel_scale"]
        single_obs[10:13] = self.gravity_orientation(quat)
        single_obs[13: 13 + n_joints] = (qj - padded_defaults) * cfg["dof_pos_scale"]
        single_obs[13 + n_joints: 13 + 2 * n_joints] = dqj * cfg["dof_vel_scale"]
        single_obs[13 + 2 * n_joints: 13 + 2 * n_joints + 15] = self.action
        return single_obs

    def set_observation(self, observation):
        self.observation = observation
        single_obs = self.compute_observation(observation)
        self.obs_history.append(single_obs)
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))
        for i, hist_obs in enumerate(self.obs_history):
            self.obs_buffer[i * 86: (i + 1) * 86] = hist_obs

    def get_action(self):
        sess = self.policy_1 if np.linalg.norm(self.cmd) < 0.05 else self.policy_2
        inp = self.obs_buffer[None, :].astype(np.float32)
        self.action = sess.run(None, {sess.get_inputs()[0].name: inp})[0].squeeze()
        if self.use_policy_action:
            cmd_q = self.action * self.config["action_scale"] + self.config["default_angles"]
        else:
            cmd_q = self.observation["q"][GROUP["lower_body"]]
        return cmd_q


class DwbcController:
    """G1Env.observe + G1DecoupledWholeBodyPolicy + G1Env.queue_action (with JointSafetyMonitor)."""

    def __init__(self, wbc_cfg, upper_body_target):
        """upper_body_target: dict joint name -> target for the 14 arm and 14 Dex3 joints."""
        self.lower = GearWbcPolicyNp(load_policy_config(wbc_cfg))
        # Upper body: InterpolationPolicy holding a single waypoint (target_time <= first tick),
        # so get_action returns this pose at every tick.
        self.upper_q = np.array([upper_body_target[PIN_JOINTS[i]] for i in GROUP["upper_body"]])
        # JointSafetyMonitor startup ramp (arm + hand joints, 100 steps at 50 Hz)
        self.ramp_steps = int(2.0 * CONTROL_FREQUENCY)
        self.startup_counter = 0
        self.initial_positions = None
        self.startup_complete = False
        self.last_obs = None

    @staticmethod
    def observe(pub):
        # G1Env.observe: hand states are read in Dex3 SDK motor order and mapped by
        # RobotModel.get_configuration_from_actuated_joints (body in BODY_ACTUATED order).
        q = np.zeros(len(PIN_JOINTS))
        dq = np.zeros(len(PIN_JOINTS))
        q[[PIN_INDEX[n] for n in BODY_ACTUATED]] = pub["body_q"]
        dq[[PIN_INDEX[n] for n in BODY_ACTUATED]] = pub["body_dq"]
        for s in ("left", "right"):
            idx = [PIN_INDEX[n] for n in HAND_ACTUATED[s]]
            q[idx] = pub[f"{s}_hand_q"]
            dq[idx] = pub[f"{s}_hand_dq"]
        return {"q": q, "dq": dq, "floating_base_pose": pub["floating_base_pose"],
                "floating_base_vel": pub["floating_base_vel"]}

    def step(self, pub):
        obs = self.observe(pub)
        self.last_obs = obs
        self.lower.set_observation(obs)
        q = np.zeros(len(PIN_JOINTS))
        q[GROUP["upper_body"]] = self.upper_q
        lower_q = self.lower.get_action()
        q[GROUP["lower_body"]] = lower_q[: len(GROUP["lower_body"])]
        q = self.safety_ramp(obs, q)
        body_q = q[[PIN_INDEX[n] for n in BODY_ACTUATED]]
        hands = {s: q[[PIN_INDEX[n] for n in HAND_ACTUATED[s]]] for s in ("left", "right")}
        return body_q, hands["left"], hands["right"], q

    def safety_ramp(self, obs, q):
        # JointSafetyMonitor.get_safe_action
        if not self.startup_complete:
            if self.initial_positions is None:
                self.initial_positions = obs["q"].copy()
            if self.startup_counter < self.ramp_steps:
                ramp = self.startup_counter / self.ramp_steps
                idx = GROUP["arms_and_hands"]
                q = q.copy()
                q[idx] = self.initial_positions[idx] + ramp * (q[idx] - self.initial_positions[idx])
                self.startup_counter += 1
            else:
                self.startup_complete = True
        return q

    def velocity_violations(self, obs):
        # JointSafetyMonitor.check_safety (velocity part): arms 6 rad/s, hands 50 rad/s.
        # In sim (env_type "sim") a violation does not shut down; it is only reported here.
        out = []
        for i in GROUP["arms_and_hands"]:
            lim = 50.0 if "_hand_" in PIN_JOINTS[i] else 6.0
            if abs(obs["dq"][i]) > lim:
                out.append((PIN_JOINTS[i], float(obs["dq"][i])))
        return out
