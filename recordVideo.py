import os
import cv2 as cv
import numpy as np
def record_video(filename="output.avi", duration=5):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot access webcam")
        return

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = 20
    out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_count = 0
    max_frames = duration * fps

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv.imshow("Recording...", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()


def denseOpticalFlow_on_video(filepath="output.avi"):
    cap = cv.VideoCapture(filepath)
    ret, frame1 = cap.read()
    if not ret:
        print("Video read error.")
        return

    prevGray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        nextGray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2,
                                           flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('Dense Optical Flow', bgr)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

        prevGray = nextGray

    cap.release()
    cv.destroyAllWindows()

def lucasKanade_on_video(filepath="output.avi"):
    cap = cv.VideoCapture(filepath)
    shiTomasiCornerParams = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lucasKanadeParams = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    ret, firstFrame = cap.read()
    if not ret:
        print("Video read error.")
        return

    prevGray = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
    prevPts = cv.goodFeaturesToTrack(prevGray, mask=None, **shiTomasiCornerParams)
    mask = np.zeros_like(firstFrame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        nextPts, status, _ = cv.calcOpticalFlowPyrLK(prevGray, gray, prevPts, None, **lucasKanadeParams)

        if nextPts is not None:
            goodNew = nextPts[status == 1]
            goodOld = prevPts[status == 1]

            for i, (new, old) in enumerate(zip(goodNew, goodOld)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % 100].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i % 100].tolist(), -1)

            img = cv.add(frame, mask)
            cv.imshow('Lucas-Kanade Optical Flow', img)

            prevGray = gray.copy()
            prevPts = goodNew.reshape(-1, 1, 2)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    # Save inside "videos/" folder
    video_path = os.path.join(os.getcwd(), "videos", "my_video.avi")
    record_video(filename=video_path, duration=10)
    lucasKanade_on_video(video_path)
    denseOpticalFlow_on_video(video_path)
