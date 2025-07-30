import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def lucasKanade_webcam():
    videoCapObj = cv.VideoCapture(1)

    shiTomasiCornerParams = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lucasKanadeParams = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    randomColors = np.random.randint(0, 255, (100, 3))

    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Failed to read from webcam.")
        return

    frameGrayPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)
    cornerPrev = cv.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)
    mask = np.zeros_like(frameFirst)

    while True:
        ret, frame = videoCapObj.read()
        if not ret:
            break

        frameGrayCur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cornersCur, foundStatus, _ = cv.calcOpticalFlowPyrLK(
            frameGrayPrev, frameGrayCur, cornerPrev, None, **lucasKanadeParams)

        if cornersCur is not None:
            cornersMatchedCur = cornersCur[foundStatus == 1]
            cornersMatchedPrev = cornerPrev[foundStatus == 1]

            for i, (cur, prev) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
                xCur, yCur = cur.ravel()
                xPrev, yPrev = prev.ravel()
                mask = cv.line(mask, (int(xCur), int(yCur)), (int(xPrev), int(yPrev)), randomColors[i % 100].tolist(), 2)
                frame = cv.circle(frame, (int(xCur), int(yCur)), 5, randomColors[i % 100].tolist(), -1)

            img = cv.add(frame, mask)
            cv.imshow('Webcam Optical Flow - Lucas Kanade', img)

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

        frameGrayPrev = frameGrayCur.copy()
        cornerPrev = cornersMatchedCur.reshape(-1, 1, 2)

    videoCapObj.release()
    cv.destroyAllWindows()

lucasKanade_webcam()