import numpy as np
import mediapipe as mp

class MediaPipeTracker:
    def __init__(self, model_complexity=1, det_conf=0.9, trk_conf=0.9, ema_alpha=1.0):
        self.hands = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
            max_num_hands=1
        )
        self.ema_alpha = float(ema_alpha)
        self.prev = None
        # Temporal smoothing with adaptive filtering
        self.history = []
        self.history_size = 3

    def close(self): self.hands.close()

    def track(self, bgr_img):
        h, w = bgr_img.shape[:2]
        rgb = bgr_img[:,:,::-1]
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.prev = None
            self.history.clear()
            return None

        lm = res.multi_hand_landmarks[0].landmark[8]
        x_raw = lm.x * w
        y_raw = lm.y * h

        # Add to history
        self.history.append((x_raw, y_raw))
        if len(self.history) > self.history_size:
            self.history.pop(0)

        # Temporal median filter to reduce jitter from detection variance
        if len(self.history) >= 2:
            x_vals = [pt[0] for pt in self.history]
            y_vals = [pt[1] for pt in self.history]
            x = float(np.median(x_vals))
            y = float(np.median(y_vals))
        else:
            x, y = x_raw, y_raw

        # Optional EMA smoothing on top
        if self.prev is None or self.ema_alpha >= 0.999:
            self.prev = np.array([x, y], dtype=np.float32)
        else:
            self.prev = self.ema_alpha * np.array([x, y]) + (1 - self.ema_alpha) * self.prev

        return float(self.prev[0]), float(self.prev[1])
