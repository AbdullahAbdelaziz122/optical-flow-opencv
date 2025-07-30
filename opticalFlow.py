import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def lucasKanade():
    root = os.getcwd()
    videoPath = os.path.join(root, 'videos/tennis.mp4')
    videoCapObj = cv.VideoCapture(videoPath)


    shiTomasiCornerParams = dict(
        maxCorners=20,
        qualityLevel=0.3,
        minDistance=50,
        blockSize=7
    )

    lucasKanadeParams = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    randomColors = np.random.randint(0, 255, (100, 3))
    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Failed to read video.")
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

            for i, (curCorner, prevCorner) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
                xCur, yCur = curCorner.ravel()
                xPrev, yPrev = prevCorner.ravel()
                mask = cv.line(mask, (int(xCur), int(yCur)), (int(xPrev), int(yPrev)), randomColors[i].tolist(), 2)
                frame = cv.circle(frame, (int(xCur), int(yCur)), 5, randomColors[i].tolist(), -1)

            img = cv.add(frame, mask)
            cv.imshow('Video', img)

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

        frameGrayPrev = frameGrayCur.copy()
        cornerPrev = cornersMatchedCur.reshape(-1, 1, 2)

    videoCapObj.release()
    cv.destroyAllWindows()



def denseOpticalFlow():
    # Read image
    root = os.getcwd()
    videoPath = os.path.join(root, 'videos/tennis.mp4')
    videoCapObj = cv.VideoCapture (videoPath)
    _, frameFirst = videoCapObj.read()
    imgPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)
    imgHSV = np.zeros_like(frameFirst)
    imgHSV[:, :, 1] = 255
    # Loop through each video frame
    while True:
        _, frameCur = videoCapObj.read()
        imgCur = cv.cvtColor(frameCur, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback (prev=imgPrev,
        next=imgCur, flow=None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=cv.
        OPTFLOW_FARNEBACK_GAUSSIAN)
        mag, ang = cv.cartToPolar (flow [:,:,0], flow [:,:,1])
        # OpenCV H is [0,180] so divide by 2
        imgHSV [:,:,0] = ang*180/np.pi/2
        imgHSV [:,:,2] = cv.normalize (mag, None, 0, 255, cv.
        NORM_MINMAX)
        imgBGR = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)
        cv.imshow('Video', imgBGR)
        cv.waitKey(15)
        imgPrev = imgCur

if __name__ == '__main__':
    lucasKanade()
    # denseOpticalFlow()