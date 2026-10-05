"""Python / onnxruntime port of the SONIC C++ deployment loop (gear_sonic_deploy).

Mirrors the logic of src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp for the
SONIC v1.1 model (policy/sonic_v1_1/*, observation_config.yaml of that directory) and the
V2 kinematic planner:

  * observation registry / gatherers (decoder + encoder, mode filter g1 / teleop); v1.1
    uses the robot-heading-normalized anchor orientations (orientation_mode 1 of
    GatherMotionAnchorOrientationMutiFrame) and has no root z observations
  * StateLogger history semantics (oldest-first, zero padding)
  * heading handling (init_base_quat, init_ref_data_root_rot, apply_delta_heading)
  * LocalMotionPlannerBase (context resampling, 30 Hz -> 50 Hz resampling, blending,
    replan triggers, idle re-adaptation)
  * action -> PD target mapping (kps / kds / action scale / default angles), with the
    v1.1 motor gain scaling of the ankle pitch motors (deploy.sh --motor-kp-scale 4,10=1.5
    --motor-kd-scale 4,10=1.5, docs/source/getting_started/download_models.md)

The ONNX models are the exact files the C++ stack feeds to TensorRT.
"""

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# policy_parameters.hpp
# ---------------------------------------------------------------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2
S_5020 = ARMATURE_5020 * NATURAL_FREQ**2
S_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
S_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
S_4010 = ARMATURE_4010 * NATURAL_FREQ**2
D_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
D_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
D_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
D_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
E_5020, E_7520_14, E_7520_22, E_4010 = 25.0, 88.0, 139.0, 5.0

# Hardware / MuJoCo joint order
_MOTOR_TYPES = (
    ["22", "22", "14", "22", "5020", "5020"] * 2
    + ["14", "5020", "5020"]
    + ["5020"] * 5 + ["4010"] * 2
    + ["5020"] * 5 + ["4010"] * 2
)
_S = {"22": S_7520_22, "14": S_7520_14, "5020": S_5020, "4010": S_4010}
_D = {"22": D_7520_22, "14": D_7520_14, "5020": D_5020, "4010": D_4010}
_E = {"22": E_7520_22, "14": E_7520_14, "5020": E_5020, "4010": E_4010}

G1_ACTION_SCALE = np.array([0.25 * _E[t] / _S[t] for t in _MOTOR_TYPES])
KPS = np.array([_S[t] for t in _MOTOR_TYPES], dtype=np.float64)
KDS = np.array([_D[t] for t in _MOTOR_TYPES], dtype=np.float64)
# ankle pitch/roll (4,5,10,11) and waist roll/pitch (13,14) use 2x gains
for _i in (4, 5, 10, 11, 13, 14):
    KPS[_i] *= 2.0
    KDS[_i] *= 2.0
# kps / kds are std::array<float, 29> in C++
KPS = KPS.astype(np.float32).astype(np.float64)
KDS = KDS.astype(np.float32).astype(np.float64)
# SONIC v1.1 motor gain scaling (motor_gain_scaling.cpp, MotorCommand gains are float): the
# left and right ankle pitch motors (hardware indices 4 and 10) use 1.5x Kp and Kd. The action
# scale keeps the unscaled gains.
MOTOR_GAIN_SCALE = {4: 1.5, 10: 1.5}
for _i, _s in MOTOR_GAIN_SCALE.items():
    KPS[_i] = float(np.float32(KPS[_i]) * np.float32(_s))
    KDS[_i] = float(np.float32(KDS[_i]) * np.float32(_s))

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# isaaclab_to_mujoco[mj] = isaaclab index of mujoco joint mj
ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
# mujoco_to_isaaclab[il] = mujoco index of isaaclab joint il
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
LOWER_BODY_MJ_ORDER_IN_IL = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
LOWER_BODY_IL_ORDER_IN_IL = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
WRIST_IL_ORDER_IN_IL = [23, 24, 25, 26, 27, 28]

POLICY_DIR = Path(__file__).resolve().parent.parent / "gear_sonic_deploy/policy/sonic_v1_1"

# Encoder input layout: encoder_observations of policy/sonic_v1_1/observation_config.yaml, in
# order, with the dimensions of the C++ observation registry (encoder input 1751).
ENCODER_OBS = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
ENC = {}
ENC_DIM = 0
for _name, _dim in ENCODER_OBS:
    ENC[_name] = slice(ENC_DIM, ENC_DIM + _dim)
    ENC_DIM += _dim
DEC_DIM = 994

# LocomotionMode (localmotion_kplanner.hpp)
IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3
_STATIC_MODES = {0, 4, 5, 6, 7, 9}
_BOXING_PUNCH = {11, 12, 13, 15, 16}
CRAWLING = 8


def is_static_motion_mode(mode):
    return mode in _STATIC_MODES


# ---------------------------------------------------------------------------
# math_utils.hpp (double versions, wxyz quaternions)
# ---------------------------------------------------------------------------
def quat_rotate(q, v):
    q_w, q_vec = q[0], np.asarray(q[1:4])
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([w, x, y, z])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def calc_heading(q):
    d = quat_rotate(q, [1.0, 0.0, 0.0])
    return math.atan2(d[1], d[0])


def quat_from_angle_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    q = np.array([math.cos(angle / 2), *(axis * math.sin(angle / 2))])
    return q / max(np.linalg.norm(q), 1e-12)


def calc_heading_quat(q):
    return quat_from_angle_axis(calc_heading(q), [0, 0, 1])


def calc_heading_quat_inv(q):
    return quat_from_angle_axis(-calc_heading(q), [0, 0, 1])


def euler_z_to_quat(a):
    return quat_from_angle_axis(a, [0, 0, 1])


def quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / max(np.linalg.norm(r), 1e-12)
    theta = math.acos(abs(dot))
    s = math.sin(theta)
    return (math.sin((1.0 - t) * theta) / s) * q0 + (math.sin(t * theta) / s) * q1


def quat_to_rotmat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def normalize_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), eps)


# ---------------------------------------------------------------------------
# Motion sequence (planner motion: 1 body, 29 joints in IsaacLab order)
# ---------------------------------------------------------------------------
class MotionSequence:
    CAPACITY = 1500

    def __init__(self, name="planner_motion"):
        self.name = name
        self.timesteps = 0
        self.joint_pos = np.zeros((self.CAPACITY, 29))
        self.joint_vel = np.zeros((self.CAPACITY, 29))
        self.root_pos = np.zeros((self.CAPACITY, 3))
        self.root_quat = np.tile([1.0, 0, 0, 0], (self.CAPACITY, 1))
        self.encode_mode = 0


# ---------------------------------------------------------------------------
# Kinematic planner (LocalMotionPlannerBase + TensorRT backend inputs)
# ---------------------------------------------------------------------------
class KinematicPlanner:
    MOTION_LOOK_AHEAD_STEPS = 2
    DEFAULT_HEIGHT = 0.788740
    INITIAL_RANDOM_SEED = 1234

    def __init__(self, onnx_path, providers, sess_options):
        self.sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.enabled = False
        self.initialized = False
        self.motion_available = False
        self.gen_frame = 0
        self.motion_50hz = MotionSequence("planner_gen")
        self.context = np.zeros((4, 36), dtype=np.float32)
        self.mode = 0
        self.target_vel = -1.0
        self.height = -1.0
        self.movement_direction = np.zeros(3, dtype=np.float32)
        self.facing_direction = np.zeros(3, dtype=np.float32)
        self.random_seed = 0
        self.allowed_pred_num_tokens = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)

    def update_input_tensors(self, mode, target_vel, target_height, movement_direction,
                             facing_direction, random_seed=-1):
        self.mode = mode if mode < 27 else 0
        self.target_vel = float(target_vel)
        self.height = float(target_height)
        self.movement_direction = np.asarray(movement_direction, dtype=np.float32)
        self.facing_direction = np.asarray(facing_direction, dtype=np.float32)
        if random_seed != -1:
            self.random_seed = random_seed

    def run_inference(self):
        feeds = {
            "context_mujoco_qpos": self.context[None].astype(np.float32),
            "target_vel": np.array([self.target_vel], dtype=np.float32),
            "mode": np.array([self.mode], dtype=np.int64),
            "movement_direction": self.movement_direction[None].astype(np.float32),
            "facing_direction": self.facing_direction[None].astype(np.float32),
            "random_seed": np.array([self.random_seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": self.allowed_pred_num_tokens,
            "height": np.array([self.height], dtype=np.float32),
        }
        qpos, n = self.sess.run(["mujoco_qpos", "num_pred_frames"], feeds)
        self._qpos = qpos[0]
        self._num_pred = int(n[0])

    def initialize(self, joint_positions_mj):
        self.initialized = False
        self.update_input_tensors(IDLE, -1.0, -1.0, [0, 0, 0], [1, 0, 0], self.INITIAL_RANDOM_SEED)
        self.gen_frame = 0
        for n in range(4):
            self.context[n, 0:3] = [0.0, 0.0, self.DEFAULT_HEIGHT]
            self.context[n, 3:7] = [1.0, 0.0, 0.0, 0.0]
            self.context[n, 7:] = joint_positions_mj
        self.run_inference()
        ok = self.resample_50hz()
        self.initialized = ok
        return ok

    def resample_50hz(self):
        t30 = self._num_pred
        q = self._qpos
        if t30 > 64 or np.isnan(q[:t30]).any():
            print("[planner] invalid output")
            return False
        m = self.motion_50hz
        m.timesteps = int(math.floor(t30 / 30.0 * 50))
        for f in range(m.timesteps):
            f30 = f / 50.0 * 30
            f0 = int(math.floor(f30))
            f1 = min(f0 + 1, t30 - 1)
            w0 = 1.0 - (f30 - f0)
            w1 = 1.0 - w0
            m.root_pos[f] = w0 * q[f0, 0:3] + w1 * q[f1, 0:3]
            m.root_quat[f] = quat_slerp(q[f0, 3:7], q[f1, 3:7], f30 - f0)
            m.joint_pos[f] = w0 * q[f0, 7 + MUJOCO_TO_ISAACLAB] + w1 * q[f1, 7 + MUJOCO_TO_ISAACLAB]
        m.joint_vel[: m.timesteps - 1] = (m.joint_pos[1:m.timesteps] - m.joint_pos[: m.timesteps - 1]) * 50.0
        m.joint_vel[m.timesteps - 1] = m.joint_vel[m.timesteps - 2]
        self.motion_available = True
        return True

    def update_context_from_motion(self, motion):
        if motion.timesteps == 0:
            raise RuntimeError("motion not ready")
        gen_time = self.gen_frame / 50.0
        for n in range(4):
            t = gen_time + n / 30.0
            f50 = t * 50.0
            f0 = min(int(math.floor(f50)), motion.timesteps - 1)
            f1 = min(f0 + 1, motion.timesteps - 1)
            w0 = 1.0 - (f50 - f0)
            w1 = 1.0 - w0
            self.context[n, 3:7] = quat_slerp(motion.root_quat[f0], motion.root_quat[f1], f50 - f0)
            self.context[n, 0:3] = w0 * motion.root_pos[f0] + w1 * motion.root_pos[f1]
            self.context[n, 7 + MUJOCO_TO_ISAACLAB] = w0 * motion.joint_pos[f0] + w1 * motion.joint_pos[f1]

    def update_planning(self, gen_frame, motion, mode, target_vel, target_height,
                        movement_direction, facing_direction):
        if not (self.initialized and self.enabled):
            return False
        self.update_input_tensors(mode, target_vel, target_height, movement_direction, facing_direction, -1)
        self.gen_frame = gen_frame + self.MOTION_LOOK_AHEAD_STEPS
        self.update_context_from_motion(motion)
        self.run_inference()
        return self.resample_50hz()


# ---------------------------------------------------------------------------
# Movement command (MovementState)
# ---------------------------------------------------------------------------
class MovementState:
    def __init__(self, mode=IDLE, movement=(0, 0, 0), facing=(1, 0, 0), speed=-1.0, height=-1.0):
        self.mode = int(mode)
        self.movement = np.asarray(movement, dtype=np.float64)
        self.facing = np.asarray(facing, dtype=np.float64)
        self.speed = float(speed)
        self.height = float(height)

    def copy(self):
        return MovementState(self.mode, self.movement.copy(), self.facing.copy(), self.speed, self.height)


# ---------------------------------------------------------------------------
# G1Deploy (control / planner threads, emulated synchronously)
# ---------------------------------------------------------------------------
class SonicDeploy:
    CONTROL_DT = 0.02
    PLANNER_DT = 0.1
    HIST_LEN = 10

    # idle readaptation thresholds
    K_ADAPT_TRIGGER = 0.10
    K_ADAPT_STOP = 0.05
    K_RECOVER_TRIGGER = 0.045

    def __init__(self, encoder_path, decoder_path, planner_path, num_threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, so, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, so, providers=providers)
        self.planner = KinematicPlanner(planner_path, providers, so)
        self.enc_dim = self.encoder.get_inputs()[0].shape[1]
        self.dec_dim = self.decoder.get_inputs()[0].shape[1]
        assert self.enc_dim == ENC_DIM and self.dec_dim == DEC_DIM, "not the SONIC v1.1 encoder / decoder"

        self.planner_motion = MotionSequence("planner_motion")
        self.current_motion = None
        self.current_frame = 0
        self.play = False

        self.history = []  # list of dict(base_quat, base_ang_vel, body_q, body_dq, last_action)
        self.last_action = np.zeros(29)
        self.token = np.zeros(64)

        # heading state
        self.reinitialize_heading = True
        self.init_base_quat = np.array([1.0, 0, 0, 0])
        self.delta_heading = 0.0
        self.init_ref_root_quat = np.array([1.0, 0, 0, 0])

        # movement / planner thread state
        self.movement_state = MovementState()
        self.last_movement_state = MovementState(IDLE, (0, 0, 0), (1, 0, 0), -1.0, -1.0)
        self.replan_interval_counter = 0.0

        # VR 3-point (teleop encoder mode 1) input buffers
        self.has_vr3 = False
        self.vr3_pos = np.zeros(9)
        self.vr3_orn = np.tile([1.0, 0, 0, 0], 3)

        # idle readapt
        self.idle_readapt_stored = False
        self.idle_readapt_state = "IDLE"
        self.idle_readapt_targets = np.zeros(29)

        self.motor_q_target = DEFAULT_ANGLES.copy()
        self.num_replans = 0

    # ------------------------------------------------------------------ input
    def set_movement(self, ms: MovementState):
        ms = ms.copy()
        if is_static_motion_mode(ms.mode):
            ms.speed = -1.0
        ms.facing = normalize_vector(ms.facing)
        ms.movement = normalize_vector(ms.movement)
        self.movement_state = ms

    def set_vr3(self, pos9, orn12):
        """ZMQManager: vr_position present -> VR 3-point control, encoder mode 1."""
        if not self.has_vr3 and self.current_motion is not None and self.current_motion.encode_mode >= 0:
            self.current_motion.encode_mode = 1
        self.has_vr3 = True
        self.vr3_pos = np.asarray(pos9, dtype=np.float64)
        self.vr3_orn = np.asarray(orn12, dtype=np.float64)

    def clear_vr3(self):
        if self.has_vr3 and self.current_motion is not None and self.current_motion.encode_mode >= 0:
            self.current_motion.encode_mode = 0
        self.has_vr3 = False

    def enable_planner(self):
        self.planner.enabled = True
        self.play = False

    # ---------------------------------------------------------- planner thread
    def planner_tick(self, motor_q_mj):
        pl = self.planner
        if not pl.enabled:
            pl.initialized = False
            return
        if not pl.initialized:
            self.planner_motion.timesteps = 0
            self.planner_motion.encode_mode = 0
            pl.initialize(motor_q_mj)
            return
        if self.planner_motion.timesteps == 0:
            return
        ms, last = self.movement_state, self.last_movement_state
        facing_changed = not np.array_equal(ms.facing, last.facing)
        height_changed = ms.height != last.height
        mode_changed = ms.mode != last.mode
        speed_changed = ms.speed != last.speed
        dir_changed = not np.array_equal(ms.movement, last.movement)
        static = is_static_motion_mode(ms.mode)
        self.replan_interval_counter += self.PLANNER_DT
        if ms.mode == RUN:
            interval = 0.1
        elif ms.mode == CRAWLING:
            interval = 0.2
        elif ms.mode in _BOXING_PUNCH:
            interval = 1.0
        else:
            interval = 1.0
        time_to_replan = False
        if self.replan_interval_counter >= interval - 1e-9:
            self.replan_interval_counter = 0.0
            time_to_replan = True
        need = mode_changed or facing_changed or height_changed or (
            not static and (speed_changed or dir_changed or (time_to_replan and ms.speed != 0)))
        if not need:
            return
        self.last_movement_state = ms.copy()
        ok = pl.update_planning(self.current_frame, self.planner_motion, ms.mode, ms.speed, ms.height,
                                ms.movement.astype(np.float32), ms.facing.astype(np.float32))
        if not ok:
            raise RuntimeError("planner update failed")
        self.num_replans += 1

    # --------------------------------------------------------- state logging
    def log_state(self, base_quat, base_ang_vel, motor_q_mj, motor_dq_mj):
        body_q = motor_q_mj[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
        body_dq = motor_dq_mj[MUJOCO_TO_ISAACLAB]
        self.history.append(dict(base_quat=np.asarray(base_quat, dtype=np.float64).copy(),
                                 base_ang_vel=np.asarray(base_ang_vel, dtype=np.float64).copy(),
                                 body_q=body_q, body_dq=body_dq, last_action=self.last_action.copy()))
        if len(self.history) > 64:
            self.history.pop(0)

    def _hist(self, n):
        """StateLogger::GetLatest(n, dt, newest_first=false): oldest first, zero-padded."""
        latest = self.history[-n:][::-1]
        zero = dict(base_quat=np.zeros(4), base_ang_vel=np.zeros(3), body_q=np.zeros(29),
                    body_dq=np.zeros(29), last_action=np.zeros(29))
        latest = latest + [zero] * (n - len(latest))
        return latest[::-1]

    # ---------------------------------------------------------------- heading
    def update_heading_state(self):
        m = self.current_motion
        if m is None or m.timesteps == 0:
            return
        if self.reinitialize_heading:
            self.init_base_quat = self.history[-1]["base_quat"].copy()
            self.delta_heading = 0.0
            self.reinitialize_heading = False
            self.init_ref_root_quat = m.root_quat[self.current_frame].copy()
        if self.current_frame == 0:
            self.init_ref_root_quat = m.root_quat[0].copy()

    def apply_delta_heading(self):
        q = quat_mul(calc_heading_quat(self.init_base_quat), calc_heading_quat_inv(self.init_ref_root_quat))
        if self.delta_heading != 0.0:
            q = quat_mul(euler_z_to_quat(self.delta_heading), q)
        return q

    # -------------------------------------------------------- motion gatherers
    def _target_frames(self, num_frames, step, hold_if_paused=True):
        m = self.current_motion
        frames = []
        for i in range(num_frames):
            f = self.current_frame
            if self.play or not hold_if_paused:
                f = min(f + i * step, m.timesteps - 1)
            frames.append(f)
        return frames

    def g_joint_pos(self, num_frames, step, idx=None):
        m = self.current_motion
        out = [m.joint_pos[f] if idx is None else m.joint_pos[f][idx] for f in self._target_frames(num_frames, step)]
        return np.concatenate(out)

    def g_joint_vel(self, num_frames, step, idx=None):
        m = self.current_motion
        out = []
        for f in self._target_frames(num_frames, step, hold_if_paused=False):
            v = m.joint_vel[f] if idx is None else m.joint_vel[f][idx]
            out.append(v if self.play else np.zeros_like(v))
        return np.concatenate(out)

    def g_anchor_ori(self, num_frames, step):
        """Anchor orientation relative to the robot heading (orientation_mode 1)."""
        m = self.current_motion
        robot_heading = calc_heading_quat(self.history[-1]["base_quat"])
        adh = self.apply_delta_heading()
        out = []
        for f in self._target_frames(num_frames, step):
            ref = quat_mul(adh, m.root_quat[f])
            r = quat_to_rotmat(quat_mul(quat_conj(robot_heading), ref))
            out.append(r[:, :2].reshape(-1))
        return np.concatenate(out)

    # ------------------------------------------------------------ observation
    def encoder_obs(self):
        m = self.current_motion
        buf = np.zeros(self.enc_dim)
        mode = m.encode_mode
        buf[0] = float(mode)
        if mode == 0:  # g1
            buf[ENC["motion_joint_positions_10frame_step5"]] = self.g_joint_pos(10, 5)
            buf[ENC["motion_joint_velocities_10frame_step5"]] = self.g_joint_vel(10, 5)
            buf[ENC["motion_anchor_orientation_heading_10frame_step5"]] = self.g_anchor_ori(10, 5)
        elif mode == 1:  # teleop
            buf[ENC["motion_anchor_orientation_heading"]] = self.g_anchor_ori(1, 1)
            lower = LOWER_BODY_MJ_ORDER_IN_IL
            buf[ENC["motion_joint_positions_lowerbody_10frame_step5"]] = self.g_joint_pos(10, 5, lower)
            buf[ENC["motion_joint_velocities_lowerbody_10frame_step5"]] = self.g_joint_vel(10, 5, lower)
            buf[ENC["vr_3point_local_target"]] = self.vr3_pos
            buf[ENC["vr_3point_local_orn_target"]] = self.vr3_orn
        else:
            raise NotImplementedError("smpl mode not used here")
        return buf

    def decoder_obs(self):
        h = self._hist(self.HIST_LEN)
        grav = [quat_rotate(quat_conj(e["base_quat"]), [0.0, 0.0, -1.0]) for e in h]
        return np.concatenate([
            self.token,
            np.concatenate([e["base_ang_vel"] for e in h]),
            np.concatenate([e["body_q"] for e in h]),
            np.concatenate([e["body_dq"] for e in h]),
            np.concatenate([e["last_action"] for e in h]),
            np.concatenate(grav),
        ])

    # ---------------------------------------------------------- control tick
    def control_tick(self, base_quat, base_ang_vel, motor_q_mj, motor_dq_mj):
        """One 50 Hz control iteration. Returns motor q targets (MuJoCo order)."""
        self.log_state(base_quat, base_ang_vel, motor_q_mj, motor_dq_mj)
        if self.current_motion is not None and self.current_motion.timesteps > 0:
            self.update_heading_state()
            enc_in = self.encoder_obs().astype(np.float32)[None]
            self.token = self.encoder.run(None, {"obs_dict": enc_in})[0][0].astype(np.float64)
            dec_in = self.decoder_obs().astype(np.float32)[None]
            action = self.decoder.run(None, {"obs_dict": dec_in})[0][0].astype(np.float64)
            self.last_action = action.copy()
            self.motor_q_target = (DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE).astype(
                np.float32).astype(np.float64)
        self.current_frame_advancement(motor_q_mj)
        return self.motor_q_target

    def current_frame_advancement(self, motor_q_mj):
        pl = self.planner
        if not (pl.enabled and pl.initialized):
            return
        pm, gen = self.planner_motion, pl.motion_50hz
        if pl.motion_available and gen.timesteps > 0:
            pl.motion_available = False
            first, success = False, False
            if pm.timesteps == 0:
                n = gen.timesteps
                pm.joint_pos[:n] = gen.joint_pos[:n]
                pm.joint_vel[:n] = gen.joint_vel[:n]
                pm.root_pos[:n] = gen.root_pos[:n]
                pm.root_quat[:n] = gen.root_quat[:n]
                pm.timesteps = n
                success = first = True
            else:
                fgen = pl.gen_frame
                new_len = fgen - self.current_frame + gen.timesteps
                if new_len > 0:
                    blend_start = max(0, fgen - self.current_frame)
                    # in-place, ascending f (same read-after-write behaviour as the C++ loop)
                    for f in range(new_len):
                        f_old = min(max(f + self.current_frame, 0), pm.timesteps - 1)
                        f_new = min(max(f + self.current_frame - fgen, 0), gen.timesteps - 1)
                        w = min(max((f - blend_start) / 8.0, 0.0), 1.0)
                        pm.joint_pos[f] = (1 - w) * pm.joint_pos[f_old] + w * gen.joint_pos[f_new]
                        pm.joint_vel[f] = (1 - w) * pm.joint_vel[f_old] + w * gen.joint_vel[f_new]
                        pm.root_pos[f] = (1 - w) * pm.root_pos[f_old] + w * gen.root_pos[f_new]
                        pm.root_quat[f] = quat_slerp(pm.root_quat[f_old], gen.root_quat[f_new], w)
                    pm.timesteps = new_len
                    success = True
            if success:
                self.current_frame = 0
                self.current_motion = pm
                self.idle_readapt_stored = False
                if first:
                    self.reinitialize_heading = True
        if self.current_motion is pm and pm.timesteps > 0 and self.play:
            new_frame = self.current_frame + 1
            if new_frame >= pm.timesteps:
                new_frame = pm.timesteps - 1
                if self.movement_state.mode == IDLE:
                    self._idle_readapt(new_frame, motor_q_mj)
            self.current_frame = new_frame

    def _idle_readapt(self, f, motor_q_mj):
        pm = self.planner_motion
        idx = LOWER_BODY_IL_ORDER_IN_IL
        if not self.idle_readapt_stored:
            self.idle_readapt_targets[idx] = pm.joint_pos[f][idx]
            self.idle_readapt_stored = True
            self.idle_readapt_state = "IDLE"
        actual = motor_q_mj[MUJOCO_TO_ISAACLAB[idx]]
        err = float(np.mean(np.abs(pm.joint_pos[f][idx] - actual)))
        s = self.idle_readapt_state
        if s == "IDLE":
            if err > self.K_ADAPT_TRIGGER:
                s = "ADAPTING"
            elif err < self.K_RECOVER_TRIGGER:
                s = "RECOVERING"
        elif s == "ADAPTING":
            if err < self.K_ADAPT_STOP:
                s = "IDLE"
        elif s == "RECOVERING":
            if err > self.K_ADAPT_TRIGGER:
                s = "ADAPTING"
        self.idle_readapt_state = s
        if s == "ADAPTING":
            pm.joint_pos[f][idx] = 0.98 * pm.joint_pos[f][idx] + 0.02 * actual
        elif s == "RECOVERING":
            pm.joint_pos[f][idx] = 0.98 * pm.joint_pos[f][idx] + 0.02 * self.idle_readapt_targets[idx]
