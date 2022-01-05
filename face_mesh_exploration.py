import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np

def set_res(cap, x,y):
    cap.set(3, int(x))
    cap.set(4, int(y))
    return str(cap.get(3)),str(cap.get(4))

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

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
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            face_image = np.zeros((100, 100),np.uint8)

            # print(image.shape)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # if results.multi_face_landmarks:
            #     for face_landmarks in results.multi_face_landmarks:
            #         mp_drawing.draw_landmarks(
            #             image=image,
            #             landmark_list=face_landmarks,
            #             connections=mp_face_mesh.FACEMESH_TESSELATION,
            #             landmark_drawing_spec=None,
            #             connection_drawing_spec=mp_drawing_styles
            #                 .get_default_face_mesh_tesselation_style())
            #         mp_drawing.draw_landmarks(
            #             image=image,
            #             landmark_list=face_landmarks,
            #             connections=mp_face_mesh.FACEMESH_CONTOURS,
            #             landmark_drawing_spec=None,
            #             connection_drawing_spec=mp_drawing_styles
            #                 .get_default_face_mesh_contours_style())
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                h, w, c = image.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                
                face_mask = np.zeros(image.shape,np.uint8)
                face_mask[cy_min:cy_max,cx_min:cx_max] = image[cy_min:cy_max,cx_min:cx_max]

                face_image = face_mask[cy_min:cy_max,cx_min:cx_max]

                face_image_YCrCb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCR_CB)
                skin_region_YCrCb = cv2.inRange(face_image_YCrCb, min_YCrCb, max_YCrCb)
                face_image = cv2.bitwise_and(face_image, face_image, mask = skin_region_YCrCb)

                # cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
            else:
                face_image = np.zeros((100, 100),np.uint8)

            image = cv2.flip(image, 1)
            face_image = cv2.flip(face_image, 1)

            # Naively calculate framerate
            current_time = datetime.now()
            dt = (current_time - start_time).microseconds / 1e6
            start_time = current_time
            if dt > 0:
                fps = int(1.0 / dt)
                cv2.putText(image, f"FPS: {fps}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

            # Flip the image horizontally for a selfie-view display.
            
            cv2.imshow('Just the face', face_image)
            cv2.imshow('Full camera', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    main()