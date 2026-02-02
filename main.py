import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque

video_path ="SampleSideonSideBatting.mov"

out_vid = "overlay_video.mp4"
out_csv = "keypoint_metrics.csv"
SMOTH_WIN = 5

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

def cal_ang(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180.0 / np.pi)

    if ang > 180:
        ang = 360 - ang
    return ang

buff = {
    "elbow": deque(maxlen=SMOTH_WIN),
    "knee": deque(maxlen=SMOTH_WIN),
    "trunk": deque(maxlen=SMOTH_WIN)
}

def smth_val(k, v):
    buff[k].append(v)
    return sum(buff[k]) / len(buff[k])

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("video not found, check folder")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_vid, fourcc, fps, (wid, hgt))

rows = []
frm_id = 0
tmp = 0

print("starting analysis...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frm_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            sh = [lm[11].x, lm[11].y]
            el = [lm[13].x, lm[13].y]
            wr = [lm[15].x, lm[15].y]

            hp = [lm[23].x, lm[23].y]
            kn = [lm[25].x, lm[25].y]
            an = [lm[27].x, lm[27].y]

            e_ang = smth_val("elbow", cal_ang(sh, el, wr))
            k_ang = smth_val("knee", cal_ang(hp, kn, an))
            t_ang = smth_val("trunk", cal_ang([sh[0], 0], sh, hp))

            rows.append({
                "frame": frm_id,
                "elbow_angle": e_ang,
                "knee_angle": k_ang,
                "trunk_lean": t_ang
            })

            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Elbow {int(e_ang)}deg", (30, 60),
                        1, 2, (0, 255, 0), 2)
            cv2.putText(frame, f"Knee {int(k_ang)}deg", (30, 110),
                        1, 2, (255, 255, 0), 2)
            cv2.putText(frame, f"Trunk {int(t_ang)}deg", (30, 160),
                        1, 2, (0, 255, 255), 2)

        writer.write(frame)

        if frm_id % 50 == 0:
            print("done frames:", frm_id)

finally:
    cap.release()
    writer.release()
    pose.close()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("saved:", out_vid)
    print("saved:", out_csv)
