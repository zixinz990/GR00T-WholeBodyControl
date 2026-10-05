"""SONIC standing test from a given whole-body pose (headless, no video).

The robot starts in the given standing pose (no elastic band, no init ramp) and SONIC runs
for a fixed duration with:
  * the kinematic planner in IDLE (zero velocity) for the legs, and
  * the upper-body override of the C++ deploy (PLANNER_FROZEN_UPPER_BODY): the 17 upper-body
    joints (waist + both arms, IsaacLab order) replace the planner reference in every future
    frame of the g1-mode encoder observation, with zero reference velocities
    (g1_deploy_onnx_ref.cpp GatherMotionJointPositionsMultiFrame /
    GatherMotionJointVelocitiesMultiFrame, policy_parameters.hpp
    upper_body_joint_isaaclab_order_in_isaaclab_index).
Dex3 finger PD targets are set to the given finger pose (native finger gains).

Everything else (scene, 200 Hz physics, PD gains, effort limits, controller port) is reused
from run_sonic_mujoco.py / sonic_deploy_py.py.

Usage (from repo root):
    source sonic_mujoco_test/cache_env.sh && source .venv_sim/bin/activate
    python sonic_mujoco_test/run_sonic_stand_pose.py --pose <stand_pose.json> --log <out.npz>
"""

import argparse
import json
import math

import mujoco
import numpy as np

from run_sonic_mujoco import (
    CONTROL_DT,
    DECIMATION,
    HAND_KD,
    HAND_KP,
    PLANNER,
    POLICY_DIR,
    SCENE,
    SIM_DT,
    G1Sim,
)
from sonic_deploy_py import (
    IDLE,
    MUJOCO_TO_ISAACLAB,
    MovementState,
    SonicDeploy,
    quat_rotate,
)

# policy_parameters.hpp
UPPER_BODY_IL_IN_IL = np.array([2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
UPPER_BODY_IL_IN_MJ = np.array([12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
assert np.array_equal(MUJOCO_TO_ISAACLAB[UPPER_BODY_IL_IN_IL], UPPER_BODY_IL_IN_MJ)


class StandPoseSim(G1Sim):
    """G1Sim that starts in a given pose, without elastic band, with given finger targets."""

    def __init__(self, body_q_mj, hand_q):
        self._init_body_q = np.asarray(body_q_mj, dtype=np.float64)
        self._init_hand_q = np.asarray(hand_q, dtype=np.float64)
        self.hand_target = self._init_hand_q.copy()
        super().__init__()
        self.q_target = self._init_body_q.copy()

    def _reset(self):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[0:3] = [0, 0, 1.0]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.body_qadr] = self._init_body_q
        d.qpos[self.hand_qadr] = self._init_hand_q
        mujoco.mj_forward(self.m, d)
        # same sole placement as G1Sim._reset: lowest foot collision box corner 1 mm above floor
        lowest = np.inf
        for g in range(self.m.ngeom):
            if self.m.geom_contype[g] and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and \
                    "ankle_roll" in self.m.body(self.m.geom_bodyid[g]).name:
                R = d.geom_xmat[g].reshape(3, 3)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            corner = d.geom_xpos[g] + R @ (self.m.geom_size[g] * [sx, sy, sz])
                            lowest = min(lowest, corner[2])
        d.qpos[2] -= lowest - 0.001
        mujoco.mj_forward(self.m, d)
        self.band = None  # no elastic band

    def step(self):
        d = self.d
        d.xfrc_applied[self.pelvis] = 0.0
        tau = np.zeros(self.m.nu)
        tau[self.body_act] = self.kp * (self.q_target - d.qpos[self.body_qadr]) - self.kd * d.qvel[self.body_vadr]
        tau[self.hand_act] = HAND_KP * (self.hand_target - d.qpos[self.hand_qadr]) - HAND_KD * d.qvel[self.hand_vadr]
        d.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
        mujoco.mj_step(self.m, d)


class SonicDeployUpperBody(SonicDeploy):
    """SonicDeploy + C++ upper-body override (has_upper_body_data_) for the g1 encoder mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ub_pos = None            # 17 values, IsaacLab order
        self.ub_vel = np.zeros(17)    # no upper_body_velocity sent -> zeros

    def set_upper_body(self, pos17_il, vel17_il=None):
        self.ub_pos = np.asarray(pos17_il, dtype=np.float64)
        self.ub_vel = np.zeros(17) if vel17_il is None else np.asarray(vel17_il, dtype=np.float64)

    def g_joint_pos(self, num_frames, step, idx=None):
        out = super().g_joint_pos(num_frames, step, idx)
        if idx is None and self.ub_pos is not None:  # only the all-joint (g1 mode) gatherer
            out = out.reshape(num_frames, 29).copy()
            out[:, UPPER_BODY_IL_IN_IL] = self.ub_pos
            out = out.reshape(-1)
        return out

    def g_joint_vel(self, num_frames, step, idx=None):
        out = super().g_joint_vel(num_frames, step, idx)
        if idx is None and self.ub_pos is not None and self.play:  # zeros when not playing
            out = out.reshape(num_frames, 29).copy()
            out[:, UPPER_BODY_IL_IN_IL] = self.ub_vel
            out = out.reshape(-1)
        return out


def robot_self_contacts(m, d):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 == 0 or b2 == 0:
            continue
        out.append([m.body(b1).name, m.body(b2).name, float(c.dist)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="stand_pose.json (body: 29 joints, dex3_fist: 14 joints)")
    ap.add_argument("--log", required=True, help="output .npz")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    pose = json.load(open(args.pose))
    probe = mujoco.MjModel.from_xml_path(str(SCENE))
    names = [probe.joint(i).name for i in range(probe.njnt)]
    body_names = [n for n in names if any(k in n for k in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"))]
    hand_names = [n for n in names if "hand" in n]
    body_q = np.array([pose["body"][n] for n in body_names])
    hand_q = np.array([pose["dex3_fist"][n] for n in hand_names])

    sim = StandPoseSim(body_q, hand_q)
    assert [sim.m.joint(j).name for j in sim.body_jid] == body_names
    ctrl = SonicDeployUpperBody(str(POLICY_DIR / "model_encoder.onnx"), str(POLICY_DIR / "model_decoder.onnx"),
                                str(PLANNER), num_threads=args.threads)
    # The C++ InitControl ramps to default_angles before the policy starts; here the robot starts
    # directly in the given pose, so the pre-policy PD target is that pose.
    ctrl.motor_q_target = body_q.copy()
    ub17 = body_q[UPPER_BODY_IL_IN_MJ]  # 17 upper-body joints in IsaacLab order
    ctrl.set_upper_body(ub17)
    ctrl.enable_planner()

    arm_mj = list(range(15, 29))
    arm_names = [body_names[i] for i in arm_mj]
    n_ticks = int(round(args.duration / CONTROL_DT))
    ts, qpos, arm_tgt, contacts = [0.0], [sim.d.qpos.copy()], [], [robot_self_contacts(sim.m, sim.d)]
    fell_t = None
    for k in range(n_ticks + 1):
        base_quat, gyro, q_mj, dq_mj = sim.robot_state()
        ctrl.set_movement(MovementState(IDLE, (0, 0, 0), (1.0, 0.0, 0.0), -1.0, -1.0))
        if ctrl.current_motion is ctrl.planner_motion and not ctrl.play:
            ctrl.play = True
        if k % 5 == 0:
            ctrl.planner_tick(q_mj)
        sim.q_target = ctrl.control_tick(base_quat, gyro, q_mj, dq_mj)
        arm_tgt.append(sim.q_target[arm_mj].copy())
        if k == n_ticks:  # command at t = duration is recorded but not simulated
            break
        for _ in range(DECIMATION):
            sim.step()
        ts.append(round((k + 1) * CONTROL_DT, 6))
        qpos.append(sim.d.qpos.copy())
        contacts.append(robot_self_contacts(sim.m, sim.d))
        if fell_t is None and sim.d.qpos[2] < 0.45:
            fell_t = ts[-1]

    qpos = np.array(qpos)
    meta = dict(
        controller="C1 SONIC (gear_sonic_deploy SONIC v1.1 model, Python port sonic_deploy_py.py)",
        sim_dt=SIM_DT, control_dt=CONTROL_DT, decimation=DECIMATION, physics_timestep_in_model=float(sim.m.opt.timestep),
        input_mode="planner IDLE (g1 encoder mode 0) + C++ upper-body override (PLANNER_FROZEN_UPPER_BODY)",
        commands=dict(planner=dict(mode="IDLE", movement=[0, 0, 0], facing=[1, 0, 0], speed=-1.0, height=-1.0),
                      upper_body_position_il17=ub17.tolist(), upper_body_velocity_il17=[0.0] * 17,
                      finger_targets=dict(zip(hand_names, hand_q.tolist()))),
        initial_state="stand_pose.json body + dex3_fist, base xy 0, identity quat, zero qvel, "
                      "lowest foot collision box corner 1 mm above floor",
        deviations_from_native_harness=[
            "starts in the given pose instead of DEFAULT_ANGLES; no elastic band (harness releases it at 0.3 s)",
            "pre-policy PD target (first tick, before the planner motion exists) = initial pose instead of "
            "DEFAULT_ANGLES (C++ InitControl would have ramped to default_angles beforehand)",
            "upper-body override ported from C++ (not present in the Python harness)",
            "Dex3 finger PD targets = fist instead of 0 (gains unchanged: kp 1.5, kd 0.1)",
            "planner facing fixed to +x (robot starts with identity heading); no video",
        ],
        fell=fell_t is not None, fell_time=fell_t,
    )
    np.savez(args.log, t=np.array(ts), qpos=qpos, xml=str(SCENE), arm_joint_names=np.array(arm_names),
             arm_pd_target=np.array(arm_tgt), self_contacts=json.dumps(contacts), meta=json.dumps(meta))

    # ---- summary
    q_end = qpos[-1][sim.body_qadr]
    tilt = math.degrees(math.acos(np.clip(quat_rotate(qpos[-1][3:7], [0, 0, 1.0])[2], -1, 1)))
    torso = sim.m.body("torso_link").id
    d2 = mujoco.MjData(sim.m)
    d2.qpos[:] = qpos[-1]
    mujoco.mj_kinematics(sim.m, d2)
    zt = d2.xmat[torso].reshape(3, 3)[:, 2]
    torso_tilt = math.degrees(math.acos(np.clip(zt[2], -1, 1)))
    pairs = sorted({(a, b) for step in contacts for a, b, _ in step})
    print(f"fell={fell_t is not None} ({fell_t})  replans={ctrl.num_replans}")
    print(f"pelvis z: t0 {qpos[0][2]:.4f}  t_end {qpos[-1][2]:.4f}   base xy drift {np.linalg.norm(qpos[-1][:2] - qpos[0][:2]) * 100:.2f} cm")
    print(f"pelvis tilt at end {tilt:.2f} deg, torso tilt at end {torso_tilt:.2f} deg")
    print(f"max |q - pose| arms {np.abs(q_end[15:29] - body_q[15:29]).max():.4f}  legs+waist {np.abs(q_end[:15] - body_q[:15]).max():.4f}")
    print(f"robot-robot contact pairs seen: {pairs}")
    print(f"saved {args.log}")


if __name__ == "__main__":
    main()
