import mediapipe as mp

class MediaPipeTracker:
    def __init__(self, model_complexity=1, det_conf=0.9, trk_conf=0.9):
        self.hands = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,  # Enables MediaPipe's internal landmark filtering
            max_num_hands=1
        )

    def close(self): self.hands.close()

    def track(self, bgr_img):
        h, w = bgr_img.shape[:2]
        rgb = bgr_img[:,:,::-1]
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None

        # Return index finger tip (landmark 8)
        # MediaPipe provides already-smoothed landmarks via min_tracking_confidence
        lm = res.multi_hand_landmarks[0].landmark[8]
        x = lm.x * w
        y = lm.y * h

        return float(x), float(y)
