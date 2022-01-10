import cv2
import sys
import mediapipe as mp
from datetime import datetime
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

from utils.face_mask import get_face_contour
from utils.fixed_list import FixedList


def set_res(cap, x, y):
    cap.set(3, int(x))
    cap.set(4, int(y))
    return str(cap.get(3)), str(cap.get(4))


def main():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    set_res(cap, 1280, 720)

    start_time = datetime.now()
    face_found = False
    frame_count = 0
    
    value_list = FixedList(256)

    # Setting up graphing
    app = pg.mkQApp("Plotting Example")

    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1000,600)
    win.setWindowTitle('pyqtgraph example: Plotting')

    pg.setConfigOptions(antialias=True)

    plot = win.addPlot(title="Updating plot")

    r_curve = plot.plot(pen="r")
    g_curve = plot.plot(pen="g")
    b_curve = plot.plot(pen="b")

    def update():
        l = value_list.get()
        t = [x[3] for x in l]
        R = [x[2] for x in l]
        G = [x[1] for x in l]
        B = [x[0] for x in l]

        r_curve.setData(R)
        g_curve.setData(G)
        b_curve.setData(B)
    
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            frame_count += 1
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
                face_found = True
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
                face_found = False
                face_image = np.zeros((200, 200), np.uint8)

            image = cv2.flip(image, 1)
            face_image = cv2.flip(face_image, 1)

            # Naively calculate framerate

            info_bar = np.zeros((100, 1280), np.uint8)

            current_time = datetime.now()
            dt = (current_time - start_time).microseconds / 1e6
            start_time = current_time
            if dt > 0:
                fps = int(1.0 / dt)
                cv2.putText(
                    info_bar, f"FPS: {fps}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
            
            try:
                cv2.imshow('Original image', image)

                if face_found:
                    cv2.imshow('Face', face_image)

                    current_R = face_image[:,:,2].sum()
                    current_G = face_image[:,:,1].sum()
                    current_B = face_image[:,:,0].sum()
                    current_t = datetime.now()

                    cv2.putText(info_bar, f"R: {current_R}", (300, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                    cv2.putText(info_bar, f"G: {current_G}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                    cv2.putText(info_bar, f"B: {current_B}", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                    value_list.push([current_R, current_G, current_B, current_t])

                cv2.imshow('Information', info_bar)
            except:
                print(sys.exc_info())

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":
    main()