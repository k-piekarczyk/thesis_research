import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np

from utils.face_mask import get_face_contour


def set_res(cap, x, y):
    cap.set(3, int(x))
    cap.set(4, int(y))
    return str(cap.get(3)), str(cap.get(4))


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    set_res(cap, 1280, 720)

    start_time = datetime.now()

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            face_mask = np.zeros(image.shape[:2], np.uint8)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            face_image = np.zeros((200, 200), np.uint8)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                h, w, c = image.shape

                face_landmarks = results.multi_face_landmarks[0]

                face_contour_points = get_face_contour(face_landmarks, w, h)

                cv2.drawContours(
                    face_mask, [face_contour_points], 0, (255, 255, 255), -1)

                min_x, min_y, br_w, br_h = cv2.boundingRect(face_contour_points)
                
                max_x = min_x + br_w
                max_y = min_y + br_h

                masked_image = cv2.bitwise_and(image, image, mask=face_mask)

                face_image = masked_image[min_y:max_y, min_x:max_x]
            else:
                face_image = np.zeros((200, 200), np.uint8)

            image = cv2.flip(image, 1)
            face_image = cv2.flip(face_image, 1)

            # Naively calculate framerate
            current_time = datetime.now()
            dt = (current_time - start_time).microseconds / 1e6
            start_time = current_time
            if dt > 0:
                fps = int(1.0 / dt)
                cv2.putText(
                    image, f"FPS: {fps}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

            cv2.imshow('Original image', image)
            cv2.imshow('Face', face_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":
    main()
