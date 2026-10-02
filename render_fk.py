import sys, os, numpy as np, cv2, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from fk_gripper_labels import GripperModel, label_episode, RAW
FF = "/home/bdursun25/.conda/envs/cuteanything/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
ep, out_dir = sys.argv[1], sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
m = GripperModel(); r = label_episode(ep, m, mount_yaw_deg=90, ext_is_cam_to_world=False)
np.savez_compressed(f"{out_dir}/{ep}_fk_gripper_labels.npz", **{f"{s}_{k}": r[s][k] for s in r["serials"] for k in ["uv", "z", "in_frame", "visible", "known"]},
                    link=m.link, points_in_link=m.pts)
caps = {s: cv2.VideoCapture(f"{RAW}/{r['rel']}/recordings/MP4/{s}.mp4") for s in r["serials"]}
T = r["T"]; raw_path = f"{out_dir}/_raw.mp4"
vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), 15 / step, (854 * 2, 520))
for t in range(T):
    panels = []
    for s in r["serials"]:
        ok, f = caps[s].read()
        if t % step: continue
        f = cv2.resize(f, (854, 480)) if ok else np.zeros((480, 854, 3), np.uint8)
        d = r[s]; nvis = 0
        for (x, y), inf, vis, kn in zip(d["uv"][t], d["in_frame"][t], d["visible"][t], d["known"][t]):
            if not inf: continue
            if vis: c = (0, 255, 0); nvis += 1
            elif kn: c = (0, 0, 255)
            else: c = (160, 160, 160)
            cv2.circle(f, (int(x), int(y)), 3 if vis else 2, c, -1, cv2.LINE_AA)
        bar = np.zeros((40, 854, 3), np.uint8)
        cv2.putText(bar, f"cam {s}  step {t}/{T}  visible {nvis}/{len(d['uv'][t])}  depth-match {d['match_rate']:.2f}", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, f]))
    if panels: vw.write(np.hstack(panels))
vw.release()
out = f"{out_dir}/{ep}_fk_gripper_both_exterior.mp4"
subprocess.run([FF, "-y", "-loglevel", "error", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", out], check=True); os.remove(raw_path)
cap = cv2.VideoCapture(out); n = int(cap.get(7))
for i, fi in enumerate([n // 5, n // 2, 4 * n // 5]):
    cap.set(1, fi); ok, f = cap.read(); cv2.imwrite(f"{out_dir}/{ep}_sample_{i}.png", f)
print("WROTE", out, n)
