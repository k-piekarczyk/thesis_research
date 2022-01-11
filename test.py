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

cap = cv2.VideoCapture('data/trimmed.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()