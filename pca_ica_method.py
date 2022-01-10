import cv2
import sys
import mediapipe as mp
from datetime import datetime, time, timedelta
import numpy as np
import traces

from scipy import signal
from sklearn.decomposition import FastICA, PCA

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

from utils.face_mask import get_face_contour
from utils.fixed_list import FixedList

FRAME_RATE = 30
FRAME_PERIOD_MS = 1000.0 / 30

TIME_SERIES_SAMPLE_LENGTH = 256
TIME_SERIES_TD = timedelta(
    milliseconds=FRAME_PERIOD_MS * TIME_SERIES_SAMPLE_LENGTH)

F1 = 0.5
F2 = 3.7

FIR = signal.firwin(32, [F1, F2], window="hamming", pass_zero=False, fs=30)

def set_res(cap, x, y):
    cap.set(3, int(x))
    cap.set(4, int(y))
    return str(cap.get(3)), str(cap.get(4))


def main():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    # set_res(cap, 1280, 720)

    start_time = datetime.now()
    face_found = False
    frame_count = 0

    value_list = FixedList(TIME_SERIES_SAMPLE_LENGTH)

    # Setting up graphing
    app = pg.mkQApp("Plotting Example")

    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1500, 600)
    win.setWindowTitle('Raw color channels')

    pg.setConfigOptions(antialias=True)

    p_r = win.addPlot(title="Red channel")
    p_r_ICA = win.addPlot(title="Red channel: ICA")
    win.nextRow()
    p_g = win.addPlot(title="Green channel")
    p_g_ICA = win.addPlot(title="Green channel: ICA")
    win.nextRow()
    p_b = win.addPlot(title="Blue channel")
    p_b_ICA = win.addPlot(title="Blue channel: ICA")

    r_curve = p_r.plot(pen="r")
    g_curve = p_g.plot(pen="g")
    b_curve = p_b.plot(pen="b")

    r_curve_ICA = p_r_ICA.plot(pen="r")
    g_curve_ICA = p_g_ICA.plot(pen="g")
    b_curve_ICA = p_b_ICA.plot(pen="b")

    def update():
        l = value_list.get()

        if len(l) > int(TIME_SERIES_SAMPLE_LENGTH/5) + 1:
            t = [x[3] for x in l]
            R = [x[0] for x in l]
            G = [x[1] for x in l]
            B = [x[2] for x in l]

            R_z = list(zip(t, R))
            G_z = list(zip(t, G))
            B_z = list(zip(t, B))

            R_ts = traces.TimeSeries(R_z)
            G_ts = traces.TimeSeries(G_z)
            B_ts = traces.TimeSeries(B_z)

            start_t = t[0]
            end_t = t[-1]

            R_resampled_ts = R_ts.sample(sampling_period=timedelta(milliseconds=FRAME_PERIOD_MS), start=start_t, end=end_t, interpolate="linear")
            G_resampled_ts = G_ts.sample(sampling_period=timedelta(milliseconds=FRAME_PERIOD_MS), start=start_t, end=end_t, interpolate="linear")
            B_resampled_ts = B_ts.sample(sampling_period=timedelta(milliseconds=FRAME_PERIOD_MS), start=start_t, end=end_t, interpolate="linear")

            R_resampled = [x[1] for x in R_resampled_ts]
            G_resampled = [x[1] for x in G_resampled_ts]
            B_resampled = [x[1] for x in B_resampled_ts]

            filtered_R = signal.lfilter(FIR, 1.0, R_resampled)[int(TIME_SERIES_SAMPLE_LENGTH/5):]
            filtered_G = signal.lfilter(FIR, 1.0, G_resampled)[int(TIME_SERIES_SAMPLE_LENGTH/5):]
            filtered_B = signal.lfilter(FIR, 1.0, B_resampled)[int(TIME_SERIES_SAMPLE_LENGTH/5):]

            r_curve.setData(R_resampled)
            g_curve.setData(G_resampled)
            b_curve.setData(B_resampled)

            pca = PCA()
            data = np.vstack((filtered_R, filtered_G, filtered_B)).reshape(-1, 3)
            comp = pca.fit_transform(data)

            # print(comp.shape)

            comp_1 = comp[:, 0]
            comp_2 = comp[:, 1]
            comp_3 = comp[:, 2]

            r_curve_ICA.setData(comp_1)
            g_curve_ICA.setData(comp_2[comp_2 < 100000])
            b_curve_ICA.setData(comp_3)
            

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

                min_x, min_y, br_w, br_h = cv2.boundingRect(
                    face_contour_points)

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
            cv2.putText(
                    info_bar, f"Samples: {len(value_list.get())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

            try:
                cv2.imshow('Original image', image)

                if face_found:
                    cv2.imshow('Face', face_image)

                    in_range = cv2.inRange(face_image, np.array([1, 1, 1]), np.array([255, 255, 255]))
                    nb_count = cv2.countNonZero(in_range)

                    current_R = face_image[:, :, 2].sum() / nb_count
                    current_G = face_image[:, :, 1].sum() / nb_count
                    current_B = face_image[:, :, 0].sum() / nb_count
                    current_t = datetime.now()

                    cv2.putText(
                        info_bar, f"R: {current_R}", (300, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                    cv2.putText(
                        info_bar, f"G: {current_G}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
                    cv2.putText(
                        info_bar, f"B: {current_B}", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                    value_list.push(
                        [current_R, current_G, current_B, current_t])

                cv2.imshow('Information', info_bar)
            except:
                print(sys.exc_info())

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":
    main()
