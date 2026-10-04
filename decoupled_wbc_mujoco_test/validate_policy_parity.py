"""Check the numpy port (dwbc_mujoco.GearWbcPolicyNp) against the repo's G1GearWbcPolicy.

Replays the per-tick observations dumped by run_dwbc_stand_pose.py --dump-policy-io through the
original class (decoupled_wbc/control/policy/g1_gear_wbc_policy.py, needs torch + onnxruntime)
and compares the stacked observation buffer and the lower-body joint targets tick by tick.
The class only needs RobotModel.get_joint_group_indices, so a small stub replaces the
pinocchio-based RobotModel.

Usage: <python with torch + onnxruntime + yaml> validate_policy_parity.py <io.npz>
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root -> import decoupled_wbc
sys.path.insert(0, str(HERE))

from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy  # noqa: E402
from dwbc_mujoco import GROUP, MODEL_PATHS, REPO, load_wbc_config  # noqa: E402


class StubRobotModel:
    def get_joint_group_indices(self, name):
        return GROUP[name]


def main():
    io = np.load(sys.argv[1])
    wbc_cfg = load_wbc_config()
    policy = G1GearWbcPolicy(StubRobotModel(), str(REPO / wbc_cfg["GEAR_WBC_CONFIG"]), ",".join(MODEL_PATHS))
    policy.use_policy_action = True
    max_obs, max_q = 0.0, 0.0
    for k in range(len(io["q"])):
        obs = {"q": io["q"][k], "dq": io["dq"][k], "floating_base_pose": io["base_pose"][k],
               "floating_base_vel": io["base_vel"][k]}
        policy.set_observation(obs)
        max_obs = max(max_obs, float(np.abs(policy.obs_buffer - io["obs_buffer"][k]).max()))
        cmd_q = policy.get_action()["body_action"][0]
        max_q = max(max_q, float(np.abs(cmd_q - io["lower_cmd_q"][k]).max()))
    print(f"ticks {len(io['q'])}: max |obs_buffer diff| = {max_obs:.3e}, max |lower cmd_q diff| = {max_q:.3e}")


if __name__ == "__main__":
    main()
