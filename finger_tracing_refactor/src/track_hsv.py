import numpy as np
import cv2

class HSVTracker:
    def __init__(self, hsv_low=(35,60,60), hsv_high=(85,255,255), morph_kernel=3, min_area=60,
                 use_flow_fallback=True, flow_win=15, flow_max_level=2,
                 use_depth_filter=True, min_depth_mm=200, max_depth_mm=600):
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.min_area = min_area
        self.use_flow = use_flow_fallback
        self.prev_gray = None
        self.prev_pt = None
        self.flow_win = flow_win
        self.flow_max_level = flow_max_level
        self.use_depth_filter = use_depth_filter
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

    def track(self, bgr_img, depth_img=None):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # Apply depth filtering if enabled and depth image provided
        if self.use_depth_filter and depth_img is not None:
            depth_mask = ((depth_img >= self.min_depth_mm) & (depth_img <= self.max_depth_mm)).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, depth_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate = None
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) >= self.min_area:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    candidate = (float(M['m10']/M['m00']), float(M['m01']/M['m00']))

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        if candidate is not None:
            self.prev_gray = gray
            self.prev_pt = candidate
            return candidate

        if self.use_flow and self.prev_gray is not None and self.prev_pt is not None:
            p0 = np.array([[self.prev_pt]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                   winSize=(self.flow_win,self.flow_win), maxLevel=self.flow_max_level)
            self.prev_gray = gray
            if st is not None and st[0,0] == 1:
                self.prev_pt = (float(p1[0,0,0]), float(p1[0,0,1]))
                return self.prev_pt

        self.prev_gray = gray
        self.prev_pt = None
        return None
