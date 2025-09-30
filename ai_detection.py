import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import base64
import time
import threading
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


def start_audio_monitoring(rms_threshold=0.003, samplerate=44100, blocksize=1024, device=None):
    """Start audio RMS monitoring in a daemon thread; sets AUDIO_CHEAT and SOUND_RMS."""
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
            # Keep audio flags off if device not available
            global AUDIO_CHEAT, SOUND_RMS
            while True:
                AUDIO_CHEAT = 0
                SOUND_RMS = 0.0
                time.sleep(1)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t


class AIDetectionEngine:
    def __init__(self, use_tracking=False, imgsz=640, conf=0.25, iou=0.45):
        # Start audio monitoring thread
        start_audio_monitoring()

        # Load YOLOv8n model
        self.model = YOLO("yolov8n.pt")  # auto-downloads weights [web:86]
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.use_tracking = bool(use_tracking)
        self.tracker_cfg = "bytetrack.yaml" if self.use_tracking else None

        # MediaPipe Face Mesh instance
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)  # high sensitivity [web:99]

        # COCO class IDs of interest
        self.COCO_PERSON = 0
        self.COCO_LAPTOP = 63
        self.COCO_CELLPHONE = 67
        self.target_ids = {self.COCO_PERSON, self.COCO_LAPTOP, self.COCO_CELLPHONE}

        # Suspicion configuration
        self.suspicious_labels = {"laptop", "cell phone"}  # from model.names mapping [web:126]

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
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None

        # Use landmarks similar to the provided logic
        face_ids = [33, 263, 1, 61, 291, 199]
        img_h, img_w, _ = roi.shape
        face_2d, face_3d = [], []
        for lm_idx, lm in enumerate(res.multi_face_landmarks[0].landmark):
            if lm_idx in face_ids:
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

        ok, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0] * 360.0)  # x
        yaw = float(angles[1] * 360.0)    # y
        return {"pitch": pitch, "yaw": yaw}

    @staticmethod
    def _classify_head_state(pitch, yaw):
        if yaw < -5:
            return "Looking Left"
        if yaw > 5:
            return "Looking Right"
        if pitch < -3:
            return "Looking Down"
        if pitch > 3:
            return "Looking Up"
        return "Forward"

    def check_frame(self, image_data: str):
        try:
            frame = self._b64_to_bgr(image_data)
            if frame is None or frame.size == 0:
                return {"error": "Invalid image data"}

            # YOLO inference
            if self.use_tracking:
                results = self.model.track(
                    source=frame, persist=True, verbose=False, tracker=self.tracker_cfg,
                    imgsz=self.imgsz, conf=self.conf, iou=self.iou
                )[0]  # first result frame [web:86]
            else:
                results = self.model.predict(
                    source=frame, verbose=False, imgsz=self.imgsz, conf=self.conf, iou=self.iou
                )[0]

            boxes = results.boxes
            if boxes is None or boxes.xyxy is None:
                # No detections
                return {
                    "suspicion": 0.0,
                    "detected_class": "none",
                    "confidence": 0.0,
                    "head_pose_flag": True,        # no persons/face -> suspicious
                    "audio_flag": bool(AUDIO_CHEAT),
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
            posed_flags = []  # for suspicion

            names = results.names  # class id -> name mapping [web:126]

            for i, (b, c, conf) in enumerate(zip(xyxy, clss, confs)):
                if c not in self.target_ids:
                    continue
                x1, y1, x2, y2 = map(int, b)
                label = names.get(int(c), str(c))

                if c == self.COCO_PERSON:
                    pose = self._estimate_head_pose(frame, [x1, y1, x2, y2])
                    if pose is None:
                        head_state = "No Face"
                        posed_flags.append(True)
                    else:
                        head_state = self._classify_head_state(pose["pitch"], pose["yaw"])
                        posed_flags.append(head_state != "Forward")
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

            # Suspicion score:
            # - 0.3 if any device (laptop/phone) is detected
            # - 0.2 if any person is not "Forward" or has "No Face"
            # - 0.2 if audio activity detected
            class_suspicion = 0.3 if any(d["label"] in self.suspicious_labels for d in devices) else 0.0
            pose_suspicion = 0.2 if any(posed_flags) else 0.0
            audio_suspicion = 0.2 if AUDIO_CHEAT else 0.0

            total_suspicion = float(min(1.0, class_suspicion + pose_suspicion + audio_suspicion))

            # For legacy UI fields
            detected_class = devices[0]["label"] if devices else ("person" if persons else "none")
            top_conf = max([p["conf"] for p in persons] + [d["conf"] for d in devices], default=0.0)
            head_pose_flag = any(posed_flags) or (len(persons) == 0)

            return {
                "suspicion": total_suspicion,
                "detected_class": detected_class,
                "confidence": float(top_conf),
                "head_pose_flag": bool(head_pose_flag),
                "audio_flag": bool(AUDIO_CHEAT),
                "deepfake_flag": False,
                "predictions": [],   # kept for compatibility, unused in YOLO path
                "persons": persons,  # list with id/bbox/conf/pose/state
                "devices": devices,  # list with label/bbox/conf
                "audio_rms": float(SOUND_RMS)
            }

        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}"}


