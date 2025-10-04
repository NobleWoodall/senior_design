import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]  # first detected hand
                lm = hand.landmark[8]  # index fingertip
                x_px, y_px = int(lm.x * w), int(lm.y * h)

                # draw just one point
                cv2.circle(frame, (x_px, y_px), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"({x_px}, {y_px})", (x_px+10, y_px-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Single Point Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
