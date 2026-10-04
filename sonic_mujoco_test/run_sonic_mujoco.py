"""Headless SONIC sim2sim in MuJoCo with scripted locomotion + EE (VR 3-point) commands.

Physics matches gear_sonic/scripts/run_sim_loop.py (scene_43dof.xml, 200 Hz, PD torques
from the deploy's kp/kd, effort-limit clipping, elastic band); the controller is the Python
port of the C++ deploy loop in sonic_deploy_py.py.

Usage (from repo root):
    source sonic_mujoco_test/cache_env.sh && source .venv_sim/bin/activate
    MUJOCO_GL=egl python sonic_mujoco_test/run_sonic_mujoco.py --out sonic_mujoco_test/sonic_mujoco_demo.mp4
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from sonic_deploy_py import (
    DEFAULT_ANGLES,
    IDLE,
    KDS,
    KPS,
    SLOW_WALK,
    WALK,
    MovementState,
    SonicDeploy,
    calc_heading,
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_slerp,
)

REPO = Path(__file__).resolve().parent.parent
SCENE = REPO / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
POLICY_DIR = REPO / "gear_sonic_deploy/policy/release"
PLANNER = REPO / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"

SIM_DT = 0.005          # SIMULATE_DT in g1_29dof_sonic_model12.yaml
CONTROL_DT = 0.02       # 50 Hz
DECIMATION = int(round(CONTROL_DT / SIM_DT))
# motor_effort_limit_list (joint order: legs, waist, L arm, L hand, R arm, R hand)
EFFORT_LIMITS = np.array(
    [88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
     88.0, 50.0, 50.0,
     25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
     25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
HAND_KP, HAND_KD = 1.5, 0.1  # Dex3Hands::open()

# VR 3-point key frames (gear_sonic/utils/teleop/vis/vr3pt_pose_visualizer.py)
VR3_FRAMES = [("left_wrist_yaw_link", np.array([0.18, -0.025, 0.0])),
              ("right_wrist_yaw_link", np.array([0.18, 0.025, 0.0])),
              ("torso_link", np.array([0.0, 0.0, 0.35]))]

BODY_JOINT_KEYS = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]


class ElasticBand:
    """unitree_sdk2py_bridge.ElasticBand (holds the pelvis until released)."""

    def __init__(self, point):
        self.kp_pos, self.kd_pos, self.kp_ang, self.kd_ang = 10000, 1000, 1000, 10
        self.point = np.asarray(point, dtype=np.float64)
        self.enable = True

    def force(self, pos, quat, lin_vel, ang_vel):
        f = self.kp_pos * (self.point - pos) + self.kd_pos * (0 - lin_vel)
        rotvec = np.zeros(3)
        mujoco.mju_quat2Vel(rotvec, quat, 1.0)
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel
        return np.concatenate([f, torque])


class G1Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.m.opt.timestep = SIM_DT
        self.d = mujoco.MjData(self.m)
        names = [self.m.joint(i).name for i in range(self.m.njnt)]
        self.body_jid = [i for i, n in enumerate(names) if any(k in n for k in BODY_JOINT_KEYS)]
        self.hand_jid = [i for i, n in enumerate(names) if "hand" in n]
        assert len(self.body_jid) == 29 and len(self.hand_jid) == 14
        self.body_qadr = self.m.jnt_qposadr[self.body_jid]
        self.body_vadr = self.m.jnt_dofadr[self.body_jid]
        self.hand_qadr = self.m.jnt_qposadr[self.hand_jid]
        self.hand_vadr = self.m.jnt_dofadr[self.hand_jid]
        # actuator index per joint (joint order == actuator order in this model)
        jnt2act = {int(self.m.actuator_trnid[a, 0]): a for a in range(self.m.nu)}
        self.body_act = np.array([jnt2act[j] for j in self.body_jid])
        self.hand_act = np.array([jnt2act[j] for j in self.hand_jid])
        self.torque_limit = EFFORT_LIMITS
        self.pelvis = self.m.body("pelvis").id
        self.q_target = DEFAULT_ANGLES.copy()
        self.kp, self.kd = KPS.copy(), KDS.copy()
        self._reset()

    def _reset(self):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[0:3] = [0, 0, 1.0]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.body_qadr] = DEFAULT_ANGLES
        mujoco.mj_forward(self.m, d)
        # lowest point of the foot collision boxes -> put soles on the floor
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
        self.band = ElasticBand(d.qpos[0:3].copy())

    def step(self):
        d = self.d
        if self.band.enable:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vel, 0)
            d.xfrc_applied[self.pelvis] = self.band.force(d.xpos[self.pelvis], d.xquat[self.pelvis],
                                                          vel[3:6], vel[0:3])
        else:
            d.xfrc_applied[self.pelvis] = 0.0
        tau = np.zeros(self.m.nu)
        tau[self.body_act] = self.kp * (self.q_target - d.qpos[self.body_qadr]) - self.kd * d.qvel[self.body_vadr]
        tau[self.hand_act] = HAND_KP * (0.0 - d.qpos[self.hand_qadr]) - HAND_KD * d.qvel[self.hand_vadr]
        d.ctrl[:] = np.clip(tau, -self.torque_limit, self.torque_limit)
        mujoco.mj_step(self.m, d)

    def robot_state(self):
        d = self.d
        return (d.qpos[3:7].copy(), d.qvel[3:6].copy(),  # IMU quat (w,x,y,z), gyro (body frame)
                d.qpos[self.body_qadr].copy(), d.qvel[self.body_vadr].copy())


class VR3PointFK:
    """Pelvis-relative VR 3-point poses (L wrist, R wrist, head) from G1 joint angles."""

    def __init__(self, model, body_qadr):
        self.m, self.d = model, mujoco.MjData(model)
        self.body_qadr = body_qadr
        self.ids = [model.body(n).id for n, _ in VR3_FRAMES]

    def __call__(self, q_body_mj):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.body_qadr] = q_body_mj
        mujoco.mj_kinematics(self.m, d)
        pos, quat = [], []
        for bid, (_, off) in zip(self.ids, VR3_FRAMES):
            pos.append(d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ off)
            quat.append(d.xquat[bid].copy())
        return np.array(pos), np.array(quat)


# ---------------------------------------------------------------------------
# Command script
# ---------------------------------------------------------------------------
def arm_pose(base_q, left=None, right=None):
    """Return a MuJoCo-order joint vector with the given 7-DOF arm angles."""
    q = base_q.copy()
    q[12:15] = 0.0  # upright waist for EE key poses
    if left is not None:
        q[15:22] = left
    if right is not None:
        q[22:29] = right
    return q


L_DEFAULT = DEFAULT_ANGLES[15:22]
R_DEFAULT = DEFAULT_ANGLES[22:29]
# EE key poses are defined through FK of these arm configurations so that the requested
# wrist position/orientation pairs are kinematically consistent.
EE_KEYPOSES = {
    "both_hands_forward": dict(left=[-1.05, 0.15, 0.0, 0.45, 0.0, 0.0, 0.0],
                               right=[-1.05, -0.15, 0.0, 0.45, 0.0, 0.0, 0.0]),
    "right_hand_up": dict(left=L_DEFAULT, right=[-2.5, -0.25, 0.0, 0.25, 0.0, 0.0, 0.0]),
    "arms_spread": dict(left=[-0.15, 1.25, 0.0, 0.25, 0.0, 0.0, 0.0],
                        right=[-0.15, -1.25, 0.0, 0.25, 0.0, 0.0, 0.0]),
    "left_hand_up": dict(left=[-2.5, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0], right=R_DEFAULT),
}


def build_schedule(seed=0):
    """List of (t_start, t_end, kind, params, label)."""
    rng = np.random.default_rng(seed)
    sched = [
        (0.0, 1.0, "loco", dict(mode=IDLE), "Stand (IDLE)"),
        (1.0, 3.2, "loco", dict(mode=WALK, speed=-1.0, rel_dir=0.0), "Walk forward (WALK, default speed)"),
        (3.2, 5.2, "loco", dict(mode=SLOW_WALK, speed=0.5, rel_dir=math.pi / 2), "Side-step left (SLOW_WALK 0.5 m/s)"),
        (5.2, 7.6, "loco", dict(mode=SLOW_WALK, speed=0.5, rel_dir=math.pi), "Walk backward (SLOW_WALK 0.5 m/s)"),
        (7.6, 9.6, "loco", dict(mode=SLOW_WALK, speed=0.5, rel_dir=-math.pi / 2), "Side-step right (SLOW_WALK 0.5 m/s)"),
    ]
    # random walk: random 45-deg-binned stick direction + random turning rate per segment
    t, t_end = 9.6, 12.9
    seg_len = (t_end - t) / 3
    dirs = np.arange(8) * math.pi / 4
    dir_names = ["fwd", "fwd-left", "left", "back-left", "back", "back-right", "right", "fwd-right"]
    for _ in range(3):
        k = int(rng.integers(0, 8))
        turn = float(rng.uniform(-0.9, 0.9))
        mode = WALK if k == 0 else SLOW_WALK
        speed = -1.0 if mode == WALK else 0.5
        sched.append((t, t + seg_len, "loco",
                      dict(mode=mode, speed=speed, rel_dir=float(dirs[k]), turn_rate=turn),
                      f"Random walk: {dir_names[k]}, turn {turn:+.2f} rad/s"))
        t += seg_len
    sched.append((12.9, 13.6, "loco", dict(mode=IDLE), "Stand (IDLE)"))
    # EE (VR 3-point) commands while standing
    sched += [
        (13.6, 15.2, "ee", dict(pose="both_hands_forward", t_move=0.9), "EE: both hands forward"),
        (15.2, 16.8, "ee", dict(pose="right_hand_up", t_move=0.9), "EE: right hand up"),
        (16.8, 18.4, "ee", dict(pose="arms_spread", t_move=0.9), "EE: arms spread"),
        (18.4, 20.0, "ee", dict(pose="start", t_move=0.9), "EE: back to start pose"),
    ]
    return sched


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def add_sphere(scene, pos, rgba, radius=0.035):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def add_arrow(scene, start, end, rgba, width=0.02):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, dtype=np.float64),
                         np.asarray(end, dtype=np.float64))
    scene.ngeom += 1


def draw_text(img, lines, org=(18, 36), scale=0.8):
    y = org[1]
    for i, (txt, color) in enumerate(lines):
        s = scale if i == 0 else scale * 0.8
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 2, cv2.LINE_AA)
        y += int(36 * s + 6)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "sonic_mujoco_demo.mp4"))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--band-release", type=float, default=0.3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="optional per-tick JSON log path")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    sim = G1Sim()
    ctrl = SonicDeploy(str(POLICY_DIR / "model_encoder.onnx"), str(POLICY_DIR / "model_decoder.onnx"),
                       str(PLANNER), num_threads=args.threads)
    fk = VR3PointFK(sim.m, sim.body_qadr)
    schedule = build_schedule(args.seed)
    for s in schedule:
        print(f"  [{s[0]:5.2f}, {s[1]:5.2f})  {s[4]}")

    renderer = writer = None
    if not args.no_video:
        sim.m.vis.global_.offwidth = max(sim.m.vis.global_.offwidth, args.width)
        sim.m.vis.global_.offheight = max(sim.m.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(sim.m, height=args.height, width=args.width)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.4, 150.0, -14.0
        cam.lookat[:] = [0, 0, 0.75]
        writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error",
                                    ffmpeg_params=["-threads", "4"])
        vopt = mujoco.MjvOption()
        vopt.sitegroup[:] = 0  # hide the scene's com_marker debug site

    # Control start: planner enabled right away (zmq_manager / gamepad_manager behaviour).
    ctrl.enable_planner()
    facing_angle = 0.0
    ee_state = None  # dict(start_pos, start_quat, from_pos, from_quat, to_pos, to_quat, t0, t_move)
    log = []
    n_ticks = int(round(args.duration / CONTROL_DT))
    next_frame_t = 0.0
    wall0 = time.time()
    fell = False

    for k in range(n_ticks):
        t = k * CONTROL_DT
        seg = next(s for s in schedule if s[0] <= t + 1e-9 < s[1] or s is schedule[-1])
        kind, p, label = seg[2], seg[3], seg[4]
        base_quat, gyro, q_mj, dq_mj = sim.robot_state()

        # ------------------------------------------------------- input thread
        if kind == "loco":
            facing_angle += p.get("turn_rate", 0.0) * CONTROL_DT
            if p["mode"] == IDLE:
                ms = MovementState(IDLE, (0, 0, 0), (math.cos(facing_angle), math.sin(facing_angle), 0), -1.0, -1.0)
            else:
                a = facing_angle + p["rel_dir"]
                ms = MovementState(p["mode"], (math.cos(a), math.sin(a), 0),
                                   (math.cos(facing_angle), math.sin(facing_angle), 0), p["speed"], -1.0)
            ctrl.set_movement(ms)
        else:  # EE command (VR 3-point), planner stays in IDLE
            ctrl.set_movement(MovementState(IDLE, (0, 0, 0), (math.cos(facing_angle), math.sin(facing_angle), 0)))
            if ee_state is None:
                # entering VR_3PT: calibrate against the current (measured) robot pose
                pos0, quat0 = fk(q_mj)
                ee_state = dict(start=(pos0, quat0), cur=(pos0, quat0), seg=None)
            if ee_state["seg"] is not seg:
                if p["pose"] == "start":
                    tgt = ee_state["start"]
                else:
                    tgt = fk(arm_pose(DEFAULT_ANGLES, **EE_KEYPOSES[p["pose"]]))
                    # keep the head/torso target where it started
                    tgt = (np.vstack([tgt[0][:2], ee_state["start"][0][2:]]),
                           np.vstack([tgt[1][:2], ee_state["start"][1][2:]]))
                ee_state.update(seg=seg, frm=ee_state["cur"], to=tgt, t0=seg[0], t_move=p["t_move"])
            a = smoothstep((t - ee_state["t0"]) / ee_state["t_move"])
            fp, fq = ee_state["frm"]
            tp, tq = ee_state["to"]
            cur_p = (1 - a) * fp + a * tp
            cur_q = np.array([quat_slerp(fq[i], tq[i], a) for i in range(3)])
            ee_state["cur"] = (cur_p, cur_q)
            ctrl.set_vr3(cur_p.reshape(-1), cur_q.reshape(-1))

        # emulate keyboard/zmq handler: start playback once planner motion is active
        if ctrl.current_motion is ctrl.planner_motion and not ctrl.play:
            ctrl.play = True
        if t >= args.band_release:
            sim.band.enable = False

        # ----------------------------------------------- planner thread (10 Hz)
        if k % 5 == 0:
            ctrl.planner_tick(q_mj)
        # ----------------------------------------------- control thread (50 Hz)
        sim.q_target = ctrl.control_tick(base_quat, gyro, q_mj, dq_mj)

        # ----------------------------------------------- physics (200 Hz) + video
        for _ in range(DECIMATION):
            sim.step()
            if writer is not None and sim.d.time >= next_frame_t - 1e-9:
                next_frame_t += 1.0 / args.fps
                pel = sim.d.xpos[sim.pelvis]
                cam.lookat[:] = 0.9 * cam.lookat + 0.1 * np.array([pel[0], pel[1], 0.72])
                renderer.update_scene(sim.d, camera=cam, scene_option=vopt)
                sc = renderer.scene
                heading = calc_heading(sim.d.qpos[3:7])
                if kind == "loco" and p["mode"] != IDLE:
                    # commanded movement direction (planner frame -> world via apply_delta_heading)
                    adh = ctrl.apply_delta_heading() if ctrl.current_motion is not None else np.array([1., 0, 0, 0])
                    mv = quat_rotate(adh, ctrl.movement_state.movement)
                    mv2 = np.array([mv[0], mv[1], 0.0])
                    st = np.array([pel[0], pel[1], 0.03]) + 0.25 * mv2
                    add_arrow(sc, st, st + 0.7 * mv2, [1.0, 0.55, 0.0, 0.95], 0.04)
                if ee_state is not None and kind == "ee":
                    pq = sim.d.xquat[sim.pelvis]
                    tgt_p = ee_state["cur"][0]
                    act_p, _ = fk_world(sim, fk)
                    for i, col in ((0, [0.1, 0.9, 0.2, 0.8]), (1, [0.95, 0.2, 0.2, 0.8])):
                        add_sphere(sc, pel + quat_rotate(pq, tgt_p[i]), col, 0.045)
                        add_sphere(sc, act_p[i], [1, 1, 1, 0.9], 0.02)
                frame = renderer.render().copy()
                lines = [(f"SONIC sim2sim (MuJoCo)   t = {sim.d.time:5.2f} s", (255, 255, 255)),
                         (f"Command: {label}", (255, 210, 60))]
                if kind == "ee":
                    lines.append(("Encoder mode: teleop (VR 3-point EE targets)   "
                                  "green/red = L/R target, white = actual", (180, 230, 255)))
                else:
                    lines.append(("Encoder mode: g1 (kinematic planner)   orange arrow = commanded direction",
                                  (180, 230, 255)))
                draw_text(frame, lines)
                writer.append_data(frame)

        # ----------------------------------------------- logging
        pel = sim.d.qpos[0:3].copy()
        ref = None
        if ctrl.current_motion is not None and ctrl.current_motion.timesteps > 0:
            f = ctrl.current_frame
            ref = quat_rotate(ctrl.apply_delta_heading(), ctrl.current_motion.root_pos[f]).tolist()
        entry = dict(t=round(t + CONTROL_DT, 3), seg=label, pelvis=pel.tolist(), ref=ref,
                     frame=ctrl.current_frame, T=ctrl.current_motion.timesteps if ctrl.current_motion is not None else 0,
                     heading=calc_heading(sim.d.qpos[3:7]), mode=ctrl.movement_state.mode,
                     enc_mode=ctrl.current_motion.encode_mode if ctrl.current_motion is not None else -9)
        if kind == "ee" and ee_state is not None:
            act_rel = fk(sim.d.qpos[sim.body_qadr])[0]
            entry["ee_err"] = np.linalg.norm(act_rel[:2] - ee_state["cur"][0][:2], axis=1).tolist()
            entry["ee_move"] = np.linalg.norm(ee_state["cur"][0][:2] - ee_state["start"][0][:2], axis=1).tolist()
        log.append(entry)
        if pel[2] < 0.45 and not fell:
            fell = True
            print(f"!!! robot fell at t={t:.2f}s (pelvis z={pel[2]:.3f})")

    if writer is not None:
        writer.close()
        renderer.close()
    wall = time.time() - wall0
    print(f"Simulated {args.duration:.1f}s in {wall:.1f}s wall, replans={ctrl.num_replans}, fell={fell}")
    summarize(log, schedule)
    if args.log:
        with open(args.log, "w") as f:
            json.dump(log, f)


def fk_world(sim, fk):
    """Actual wrist key points in world frame (same offsets as the VR 3-point targets)."""
    d = sim.d
    out = []
    for bid, (_, off) in zip(fk.ids, VR3_FRAMES):
        out.append(d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ off)
    return np.array(out), None


def summarize(log, schedule):
    print("\nPer-segment summary (displacement expressed in the robot's heading frame at segment start):")
    print(f"{'segment':45s} {'fwd[m]':>8s} {'left[m]':>8s} {'dyaw[deg]':>9s} {'min z':>6s} {'EE err L/R[cm]':>15s} {'EE target shift L/R[cm]':>24s}")
    for s in schedule:
        ents = [e for e in log if s[0] < e["t"] <= s[1] + 1e-9]
        if not ents:
            continue
        p0 = np.array(ents[0]["pelvis"])
        p1 = np.array(ents[-1]["pelvis"])
        h0 = ents[0]["heading"]
        dp = p1 - p0
        fwd = dp[0] * math.cos(h0) + dp[1] * math.sin(h0)
        left = -dp[0] * math.sin(h0) + dp[1] * math.cos(h0)
        dyaw = math.degrees((ents[-1]["heading"] - h0 + math.pi) % (2 * math.pi) - math.pi)
        minz = min(e["pelvis"][2] for e in ents)
        ee = mv = ""
        if "ee_err" in ents[-1]:
            tail = 100 * np.mean([e["ee_err"] for e in ents[-10:]], axis=0)
            ee = f"{tail[0]:.1f} / {tail[1]:.1f}"
            shift = 100 * np.array(ents[-1]["ee_move"])
            mv = f"{shift[0]:.1f} / {shift[1]:.1f}"
        print(f"{s[4]:45s} {fwd:8.2f} {left:8.2f} {dyaw:9.1f} {minz:6.3f} {ee:>15s} {mv:>24s}")


if __name__ == "__main__":
    main()
