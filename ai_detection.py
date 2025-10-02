import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import base64
import time
import threading
from collections import deque

import numpy as np
import cv2

# Ultralytics YOLOv8
from ultralytics import YOLO  # pip install ultralytics

# MediaPipe for head pose
import mediapipe as mp

# Audio monitoring
import sounddevice as sd

# Globals for audio state
AUDIO_CHEAT = 0
SOUND_RMS = 0.0


def start_audio_monitoring(rms_threshold=0.015, samplerate=44100, blocksize=1024, device=None):
    """Start audio RMS monitoring; sets AUDIO_CHEAT and SOUND_RMS with a higher threshold to reduce sensitivity."""
    def audio_callback(indata, frames, time_info, status):
        global AUDIO_CHEAT, SOUND_RMS
        try:
            if indata is not None and len(indata) > 0:
                rms = float(np.sqrt(np.mean(np.square(indata[:, 0]))))
                SOUND_RMS = rms
                AUDIO_CHEAT = 1 if rms > rms_threshold else 0
        except Exception:
            AUDIO_CHEAT = 0
            SOUND_RMS = 0.0

    def runner():
        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize, device=device):
                while True:
                    time.sleep(0.1)
        except Exception:
            global AUDIO_CHEAT, SOUND_RMS
            while True:
                AUDIO_CHEAT = 0
                SOUND_RMS = 0.0
                time.sleep(1)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t


class AIDetectionEngine:
    def __init__(self, use_tracking=False, imgsz=640, conf=0.6, iou=0.65):
        # Start audio monitoring thread with higher RMS threshold
        start_audio_monitoring(rms_threshold=0.015)

        # Load YOLOv8n model with stricter thresholds to reduce false positives
        self.model = YOLO("yolov8n.pt")  # auto-downloads weights
        self.imgsz = int(imgsz)
        self.conf = float(conf)   # stricter than 0.25
        self.iou = float(iou)     # stricter than 0.45
        self.use_tracking = bool(use_tracking)
        self.tracker_cfg = "bytetrack.yaml" if self.use_tracking else None

        # MediaPipe Face Mesh instance with higher confidences and single face
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # COCO class IDs of interest
        self.COCO_PERSON = 0
        self.COCO_LAPTOP = 63
        self.COCO_CELLPHONE = 67
        self.target_ids = {self.COCO_PERSON, self.COCO_LAPTOP, self.COCO_CELLPHONE}

        # Suspicion configuration
        self.suspicious_labels = {"laptop", "cell phone"}

        # Head pose forward band (wider band = less sensitive)
        self.forward_pitch = 6.0
        self.forward_yaw = 10.0

        # Temporal smoothing and audio debounce
        self.state_hist = deque(maxlen=10)   # ~1s if ~10 fps checks
        self.audio_strikes = 0
        self.audio_strikes_needed = 0.5       # ~0.5s at 100ms audio callback

    def _b64_to_bgr(self, image_data: str):
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _estimate_head_pose(self, frame_bgr, bbox):
        # bbox [x1, y1, x2, y2]
        h, w, _ = frame_bgr.shape
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        # Ignore tiny boxes (noise)
        if (x2 - x1) * (y2 - y1) < 400:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None

        # Selected landmarks
        face_ids = [33, 263, 1, 61, 291, 199]
        img_h, img_w, _ = roi.shape
        face_2d, face_3d = [], []
        lms = res.multi_face_landmarks[0].landmark
        for idx in face_ids:
            lm = lms[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        if len(face_2d) < 6:
            return None

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        ok, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0] * 360.0)  # x
        yaw = float(angles[1] * 360.0)    # y
        return {"pitch": pitch, "yaw": yaw}

    def _classify_head_state(self, pitch, yaw):
        if yaw < -self.forward_yaw:
            return "Looking Left"
        if yaw > self.forward_yaw:
            return "Looking Right"
        if pitch < -self.forward_pitch:
            return "Looking Down"
        if pitch > self.forward_pitch:
            return "Looking Up"
        return "Forward"

    def check_frame(self, image_data: str):
        try:
            frame = self._b64_to_bgr(image_data)
            if frame is None or frame.size == 0:
                return {"error": "Invalid image data"}

            # YOLO inference with stricter thresholds
            if self.use_tracking:
                results = self.model.track(
                    source=frame, persist=True, verbose=False, tracker=self.tracker_cfg,
                    imgsz=self.imgsz, conf=self.conf, iou=self.iou
                )[0]
            else:
                results = self.model.predict(
                    source=frame, verbose=False, imgsz=self.imgsz, conf=self.conf, iou=self.iou
                )[0]

            boxes = results.boxes
            if boxes is None or boxes.xyxy is None:
                # No detections -> neutral, not auto-flagging pose
                return {
                    "suspicion": 0.0,
                    "detected_class": "none",
                    "confidence": 0.0,
                    "head_pose_flag": False,
                    "audio_flag": False,
                    "deepfake_flag": False,
                    "predictions": [],
                    "persons": [],
                    "devices": [],
                    "audio_rms": float(SOUND_RMS)
                }

            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = None
            if self.use_tracking and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)

            persons = []
            devices = []
            names = getattr(results, "names", {}) or {}

            primary_state = None

            for i, (b, c, conf) in enumerate(zip(xyxy, clss, confs)):
                if c not in self.target_ids:
                    continue
                x1, y1, x2, y2 = map(int, b)
                label = names.get(int(c), str(int(c)))

                # ignore tiny boxes
                if (x2 - x1) * (y2 - y1) < 400:
                    continue

                if c == self.COCO_PERSON:
                    pose = self._estimate_head_pose(frame, [x1, y1, x2, y2])
                    if pose is None:
                        head_state = "Forward"  # neutral if not confident
                    else:
                        head_state = self._classify_head_state(pose["pitch"], pose["yaw"])
                    if primary_state is None:
                        primary_state = head_state
                    persons.append({
                        "id": int(ids[i]) if ids is not None and i < len(ids) else None,
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "pose": pose,
                        "state": head_state
                    })
                else:
                    devices.append({
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf)
                    })

            # Temporal smoothing of head pose
            if primary_state is None:
                # no person detected -> treat as neutral
                self.state_hist.append("Forward")
            else:
                self.state_hist.append(primary_state)

            non_forward_ratio = sum(1 for s in self.state_hist if s != "Forward") / max(1, len(self.state_hist))
            pose_flag = non_forward_ratio >= 0.6

            # Debounce audio
            global AUDIO_CHEAT
            if AUDIO_CHEAT:
                self.audio_strikes = min(self.audio_strikes + 1, 1000)
            else:
                self.audio_strikes = max(self.audio_strikes - 1, 0)
            audio_active = (self.audio_strikes >= self.audio_strikes_needed)

            # Suspicion scoring with lower weights and combo gating
            device_flag = any(d["label"] in self.suspicious_labels for d in devices)

            class_suspicion = 0.2 if device_flag else 0.0
            pose_suspicion  = 0.3 if pose_flag else 0.0
            audio_suspicion = 0.2 if audio_active else 0.0

            raw_score = class_suspicion + pose_suspicion + audio_suspicion
            active_flags = (1 if device_flag else 0) + (1 if pose_flag else 0) + (1 if audio_active else 0)
            if active_flags >= 2:
                total_suspicion = float(min(1.0, raw_score + 0.3))
            else:
                total_suspicion = float(raw_score)

            detected_class = devices[0]["label"] if devices else ("person" if persons else "none")
            top_conf = max([p["conf"] for p in persons] + [d["conf"] for d in devices], default=0.0)

            return {
                "suspicion": total_suspicion,
                "detected_class": detected_class,
                "confidence": float(top_conf),
                "head_pose_flag": bool(pose_flag),
                "audio_flag": bool(audio_active),
                "deepfake_flag": False,
                "predictions": [],
                "persons": persons,
                "devices": devices,
                "audio_rms": float(SOUND_RMS)
            }

        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}"}
