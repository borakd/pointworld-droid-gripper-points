"""Label gripper points on every frame of a PointWorld-DROID episode via kinematics + depth visibility.

Inputs per episode: raw DROID trajectory.h5 (EE pose + gripper width per step), PointWorld cameras JSON
(optimized extrinsics), PointWorld flows H5 (intrinsics at 320x180), PointWorld depth H5 (per-frame depth).
Output: uv (T,N,2) at 854x480, depth_cam (T,N), in_frame (T,N), visible (T,N), visible_known (T,N).
"""
import json, sys, numpy as np, h5py, yourdfpy, trimesh
from scipy.spatial.transform import Rotation as Rot

URDF_DIR = "/home/bdursun25/.cache/robot_descriptions/robotiq_arg85_description"
PW = "/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored/droid"
RAW = "/frozen/avg/bora_data/droid_datasets/droid_raw_full_v2/droid_raw/1.0.1"
OUT_HW = (480, 854)

class GripperModel:
    def __init__(self, n_points=160, seed=0):
        fh = lambda fname: fname.replace("package://robotiq_arg85_description", URDF_DIR)
        self.urdf = yourdfpy.URDF.load(f"{URDF_DIR}/robots/robotiq_arg85_description.URDF", filename_handler=fh)
        rng = np.random.default_rng(seed)
        # sample surface points per link in the link's own frame, area-weighted across links
        meshes = {}
        for name, link in self.urdf.link_map.items():
            for vis in link.visuals:
                if vis.geometry.mesh is None: continue
                m = trimesh.load(fh(fname=vis.geometry.mesh.filename), force="mesh")
                if vis.geometry.mesh.scale is not None: m.apply_scale(vis.geometry.mesh.scale)
                if vis.origin is not None: m.apply_transform(vis.origin)
                meshes[name] = m
        areas = np.array([m.area for m in meshes.values()]); alloc = np.maximum(4, np.round(n_points * areas / areas.sum())).astype(int)
        self.pts, self.nrm, self.link = [], [], []
        for (name, m), k in zip(meshes.items(), alloc):
            p, fi = trimesh.sample.sample_surface_even(m, int(k), seed=int(rng.integers(1e9)))
            self.pts.append(p); self.nrm.append(m.face_normals[fi]); self.link += [name] * len(p)
        self.pts = np.concatenate(self.pts); self.nrm = np.concatenate(self.nrm); self.link = np.array(self.link); self.links = list(meshes)

    def points_in_base(self, finger_q):
        self.urdf.update_cfg({"finger_joint": float(finger_q)})
        out = np.empty_like(self.pts); nout = np.empty_like(self.nrm)
        for name in self.links:
            T = self.urdf.get_transform(name, self.urdf.base_link); sel = self.link == name
            out[sel] = self.pts[sel] @ T[:3, :3].T + T[:3, 3]; nout[sel] = self.nrm[sel] @ T[:3, :3].T
        return out, nout

def episode_paths(ep):
    cam = json.load(open(f"{PW}/cameras/{ep}_cameras.json"))
    rel = cam["scene_path"].split("/droid_raw/1.0.1/")[1].rstrip("/")
    return cam, f"{RAW}/{rel}/trajectory.h5", f"{rel}"

def label_episode(ep, model, mount_yaw_deg=0.0, ext_is_cam_to_world=None, depth_tol=(0.03, 0.03)):
    cam, traj_path, rel = episode_paths(ep)
    fl = h5py.File(f"{PW}/flows-fs-optimized/{ep}_flows.h5", "r"); dp = h5py.File(f"{PW}/depth_320x180/{ep}_depth.h5", "r")
    tr = h5py.File(traj_path, "r")
    cart = tr["observation/robot_state/cartesian_position"][:]; grip = tr["observation/robot_state/gripper_position"][:]
    T = len(cart)
    clip0 = sorted([k for k in fl if ":" in k], key=lambda k: int(k.split(":")[0]))[0]
    serials = [k.split("_")[1] for k in fl[clip0] if k.startswith("camera_") and k.endswith("_ext")]
    # gripper points in robot base frame for every step
    Rm = Rot.from_euler("xyz", cart[:, 3:]).as_matrix() @ Rot.from_euler("z", mount_yaw_deg, degrees=True).as_matrix()
    q = np.clip(grip, 0, 1) * 0.725
    qs = np.round(q, 3); cache = {}
    Pw = np.empty((T, len(model.pts), 3)); Nw = np.empty_like(Pw)
    for t in range(T):
        if qs[t] not in cache: cache[qs[t]] = model.points_in_base(qs[t])
        p, n = cache[qs[t]]
        Pw[t] = p @ Rm[t].T + cart[t, :3]; Nw[t] = n @ Rm[t].T
    res = {"serials": serials, "T": T, "rel": rel}
    for s in serials:
        K = fl[clip0][f"camera_{s}_ext"]["intrinsic"][()].astype(np.float64)
        E = np.array(cam[s]["optimized_extrinsics"], dtype=np.float64)
        D = dp[f"{s}+ext"]["depth"][:]; nD = D.shape[0]
        cands = [True, False] if ext_is_cam_to_world is None else [ext_is_cam_to_world]
        best = None
        for c2w in cands:
            W2C = np.linalg.inv(E) if c2w else E
            Pc = Pw @ W2C[:3, :3].T + W2C[:3, 3]
            z = Pc[..., 2]; uv = Pc[..., :2] / np.maximum(z[..., None], 1e-6) @ K[:2, :2].T + K[:2, 2]  # 320x180 pixels
            inf = (z > 0.05) & (uv[..., 0] >= 0) & (uv[..., 0] < 320) & (uv[..., 1] >= 0) & (uv[..., 1] < 180)
            # depth test (frame t of depth assumed == trajectory step t; clipped at nD)
            ui = np.clip(uv[..., 0].astype(int), 0, 319); vi = np.clip(uv[..., 1].astype(int), 0, 179)
            ti = np.minimum(np.arange(T), nD - 1)[:, None]
            obs = D[ti, vi, ui] / 1000.0
            Nc = Nw @ W2C[:3, :3].T
            front = (Nc * Pc).sum(-1) < 0             # surface normal faces the camera
            known = inf & (obs > 0)
            tol = depth_tol[0] + depth_tol[1] * z
            vis = known & front & (obs > z - tol)    # front-facing and nothing significantly in front of it
            match = known & front & (np.abs(obs - z) < tol)  # observed surface IS the gripper (calibration check)
            score = match.sum() / max((known & front).sum(), 1)
            if best is None or score > best["match_rate"]:
                best = dict(c2w=c2w, uv=uv * np.array([OUT_HW[1] / 320, OUT_HW[0] / 180]), z=z, in_frame=inf,
                            visible=vis, known=known, front=front, match_rate=float(score))
        res[s] = best
    return res

if __name__ == "__main__":
    ep = sys.argv[1]; m = GripperModel()
    for yaw in [0, 45, 90, -45]:
        r = label_episode(ep, m, mount_yaw_deg=yaw)
        print("yaw", yaw, {s: (r[s]["c2w"], round(r[s]["match_rate"], 3), round(float(r[s]["in_frame"].mean()), 3), round(float(r[s]["visible"].sum() / max(r[s]["in_frame"].sum(), 1)), 3)) for s in r["serials"]})
