import sys, os, json, glob, time, random, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fk_gripper_labels import GripperModel, label_episode, PW, RAW
cams = sorted(glob.glob(f"{PW}/cameras/*_cameras.json")); print("pointworld episodes", len(cams))
# coverage: does each PW episode have a local raw trajectory + exterior MP4s?
have = 0; missing = []
for c in cams:
    j = json.load(open(c)); rel = j["scene_path"].split("/droid_raw/1.0.1/")[1].rstrip("/")
    ok = os.path.exists(f"{RAW}/{rel}/trajectory.h5") and all(os.path.exists(f"{RAW}/{rel}/recordings/MP4/{s}.mp4") for s in j if s.isdigit())
    have += ok
    if not ok: missing.append(os.path.basename(c))
print("with local raw trajectory+mp4", have, "missing", len(missing), missing[:3])
random.seed(0); sample = random.sample(cams, 40); m = GripperModel(); rows = []
for c in sample:
    ep = os.path.basename(c).replace("_cameras.json", "")
    try:
        t0 = time.time(); r = label_episode(ep, m, mount_yaw_deg=90, ext_is_cam_to_world=False); dt = time.time() - t0
        for s in r["serials"]:
            d = r[s]; inf = d["in_frame"]
            rows.append(dict(ep=ep, cam=s, T=r["T"], sec=dt, match=d["match_rate"], frac_frames_in=float(inf.any(1).mean()),
                             frac_frames_vis=float(d["visible"].any(1).mean()), vis_of_front=float((d["visible"].sum()) / max((inf & d["front"]).sum(), 1)),
                             unknown=float((inf & ~d["known"]).sum() / max(inf.sum(), 1))))
    except Exception as e:
        rows.append(dict(ep=ep, err=repr(e)[:200]))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "sample_eval.json"), "w"), indent=1)
ok = [r for r in rows if "err" not in r]; print("errors", [r for r in rows if "err" in r])
for k in ["match", "frac_frames_in", "frac_frames_vis", "vis_of_front", "unknown", "sec", "T"]:
    v = np.array([r[k] for r in ok]); print(f"{k:16s} p10 {np.percentile(v,10):.3f}  median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}")
print("cams with match<0.4:", [(r["ep"], r["cam"], round(r["match"], 2)) for r in ok if r["match"] < 0.4])
