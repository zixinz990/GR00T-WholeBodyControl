"""Run the GR00T Decoupled WBC in MuJoCo from a given standing pose and log the state.

The robot starts on the floor (no elastic band) in the pose from --pose (JSON with "body" =
29 joint angles and "dex3_fist" = 14 finger angles). From t = 0 the lower-body RL policy
(Balance network, zero velocity command, height 0.74, rpy 0) controls legs + waist, and the
upper body is held at the pose's arm and finger angles.

Usage (from repo root):
    source decoupled_wbc_mujoco_test/cache_env.sh
    .venv_sim/bin/python decoupled_wbc_mujoco_test/run_dwbc_stand_pose.py \
        --pose <stand_pose.json> --out <log.npz> [--duration 2.0] [--dump-policy-io <io.npz>]
"""

import argparse
import json

import mujoco
import numpy as np

from dwbc_mujoco import (
    CONTROL_FREQUENCY,
    PIN_JOINTS,
    SCENE,
    DwbcController,
    DwbcSim,
    load_wbc_config,
)

ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]


def lowest_collision_point(m, d):
    """Lowest world z over all robot collision geoms (feet in a standing pose)."""
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        t = m.geom_type[g]
        size = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            z = (v @ R.T + p)[:, 2].min()
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = p[2] - size[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            z = p[2] - np.abs(R[2]) @ size
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            z = p[2] - abs(R[2, 2]) * size[1] - size[0]
        else:
            continue
        zmin = min(zmin, z)
    return zmin


def robot_contacts(m, d_scratch, qpos):
    """[body1, body2, dist] for every contact between two robot bodies at this qpos."""
    d_scratch.qpos[:] = qpos
    mujoco.mj_fwdPosition(m, d_scratch)
    out = []
    for i in range(d_scratch.ncon):
        c = d_scratch.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            continue
        out.append([m.body(b1).name, m.body(b2).name, float(c.dist)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--dump-policy-io", default=None, help="save per-tick policy inputs/outputs for the parity check")
    args = ap.parse_args()

    pose = json.load(open(args.pose))
    joint_targets = {**pose["body"], **pose["dex3_fist"]}

    wbc_cfg = load_wbc_config()
    sim = DwbcSim(wbc_cfg)
    m = sim.m
    ctrl = DwbcController(wbc_cfg, {n: joint_targets[n] for n in PIN_JOINTS[15:]})

    # initial state: identity base orientation, pose joints, feet 1 mm above the floor
    qpos = m.qpos0.copy()
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for j in range(1, m.njnt):
        qpos[m.jnt_qposadr[j]] = joint_targets[m.joint(j).name]
    scratch = mujoco.MjData(m)
    qpos[2] = 0.0
    scratch.qpos[:] = qpos
    mujoco.mj_kinematics(m, scratch)
    qpos[2] = 0.001 - lowest_collision_point(m, scratch)
    sim.reset(qpos)

    decimation = int(round(wbc_cfg["REWARD_DT"] / wbc_cfg["SIMULATE_DT"]))  # G1Env.step_simulator: 4
    n_ticks = int(round(args.duration * CONTROL_FREQUENCY))
    arm_qadr = [m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS]
    pin_arm_idx = [PIN_JOINTS.index(n) for n in ARM_JOINTS]

    ts, qposes, arm_targets, contacts, violations = [], [], [], [], []
    io = {"q": [], "dq": [], "base_pose": [], "base_vel": [], "obs_buffer": [], "lower_cmd_q": []}

    def record_state():
        ts.append(round(sim.d.time, 6))
        qposes.append(sim.d.qpos.copy())
        contacts.append(robot_contacts(m, scratch, sim.d.qpos))

    record_state()
    for k in range(n_ticks + 1):
        # sync-mode loop: observe the last published state, compute, queue the command
        body_q, lh_q, rh_q, q_full = ctrl.step(sim.published)
        arm_targets.append(q_full[pin_arm_idx].copy())
        v = ctrl.velocity_violations(ctrl.last_obs)
        if v:
            violations.append([k, v])
        if args.dump_policy_io:
            o = ctrl.last_obs
            io["q"].append(o["q"]); io["dq"].append(o["dq"])
            io["base_pose"].append(o["floating_base_pose"]); io["base_vel"].append(o["floating_base_vel"])
            io["obs_buffer"].append(ctrl.lower.obs_buffer.copy())
            io["lower_cmd_q"].append(q_full[:15].copy())
        if k == n_ticks:
            break  # the last command is logged but not applied (end of the run)
        sim.queue_command(body_q, lh_q, rh_q)
        for _ in range(decimation):
            sim.sim_step()
        record_state()

    qposes = np.array(qposes)
    meta = {
        "controller": "GR00T Decoupled WBC (decoupled_wbc): G1GearWbcPolicy lower body + upper body joint targets",
        "sim_dt": wbc_cfg["SIMULATE_DT"], "control_dt": 1.0 / CONTROL_FREQUENCY, "substeps": decimation,
        "mujoco_version": mujoco.__version__,
        "policy": "GR00T-WholeBodyControl-Balance.onnx (||cmd|| < 0.05), use_policy_action=True from t=0",
        "commands": {"navigate_cmd": [0.0, 0.0, 0.0], "height_cmd": float(ctrl.lower.height_cmd),
                     "rpy_cmd": [0.0, 0.0, 0.0], "use_teleop_policy_cmd": False,
                     "upper_body_target": "pose arms (14) + dex3_fist (14), constant from t=0"},
        "gains": "body: MOTOR_KP/MOTOR_KD in control/main/teleop/configs/g1_29dof_gear_wbc.yaml; "
                 "hands: HandCommandSender kp 1.0 (thumb_0 2.0), kd 0.2 (thumb_0 0.5); "
                 "torque clip: motor_effort_limit_list",
        "initial_base_z": float(qpos[2]),
        "safety_velocity_violations": violations,
        "deviations": [
            "No elastic band: the robot starts standing on the floor and the policy is active at t=0 "
            "(real stack: torso held by the band, policy enabled with ']', band released with '9').",
            "Policy observation history starts zero-padded at t=0 (the real stack keeps observing "
            "while the policy output is not applied, so its history is already full when enabled).",
            "Sync-mode timing (run_g1_control_loop --sim_sync_mode): 4 sim steps per 50 Hz tick, the policy "
            "observes the state published before the last sim step (5 ms old); DDS delivery delays are not "
            "modelled and the passive first step_simulator call before the first command is skipped.",
            "Upper-body InterpolationPolicy replaced by its steady state: a single waypoint equal to the "
            "pose, i.e. target_upper_body_pose published with target_time <= first tick.",
            "JointSafetyMonitor startup ramp reproduced (no-op because the initial state equals the target); "
            "velocity checks only reported (env_type sim never shuts down).",
            "G1GearWbcPolicy ported to numpy/onnxruntime (.venv_sim has no torch); see validate_policy_parity.py.",
            "Dex3 mapping reproduced as in the bridge: hand motor i (SDK order thumb0-2, index0-1, middle0-1) "
            "drives the i-th MuJoCo hand joint (thumb0-2, middle0-1, index0-1); identical index/middle "
            "targets in the fist make this swap irrelevant here.",
        ],
    }
    np.savez(
        args.out,
        t=np.array(ts),
        qpos=qposes,
        xml=str(SCENE),
        arm_joint_names=np.array(ARM_JOINTS),
        arm_pd_target=np.array(arm_targets),
        self_contacts=json.dumps(contacts),
        meta=json.dumps(meta),
    )
    if args.dump_policy_io:
        np.savez(args.dump_policy_io, **{k: np.array(v) for k, v in io.items()})

    # summary
    def tilt(qp):
        d2 = mujoco.MjData(m)
        d2.qpos[:] = qp
        mujoco.mj_kinematics(m, d2)
        zt = d2.xmat[m.body("torso_link").id].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(zt[2], -1, 1))))

    pose_vec = np.array([joint_targets[m.joint(j).name] for j in range(1, m.njnt)])
    qj_end = qposes[-1][7:]
    names = [m.joint(j).name for j in range(1, m.njnt)]
    arm_err = max(abs(qj_end[i] - pose_vec[i]) for i, n in enumerate(names) if n in ARM_JOINTS)
    lower_err = max(abs(qj_end[i] - pose_vec[i]) for i, n in enumerate(names)
                    if any(p in n for p in ("hip", "knee", "ankle", "waist")))
    fell = bool(qposes[:, 2].min() < 0.2) or tilt(qposes[-1]) > 60
    print(json.dumps({
        "fell": fell,
        "pelvis_z_t0": float(qposes[0, 2]), "pelvis_z_end": float(qposes[-1, 2]),
        "base_xy_drift": float(np.linalg.norm(qposes[-1, :2] - qposes[0, :2])),
        "torso_tilt_deg_end": tilt(qposes[-1]),
        "max_abs_arm_err_end": float(arm_err), "max_abs_leg_waist_err_end": float(lower_err),
        "n_steps_with_self_contact": sum(1 for c in contacts if c),
        "self_contact_pairs": sorted({(a, b) for c in contacts for a, b, _ in c}),
        "velocity_violations": len(violations),
    }, indent=1))


if __name__ == "__main__":
    main()
