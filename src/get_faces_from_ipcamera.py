import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--faces", default=20,
                help="Number of faces that camera will get")
ap.add_argument("--output", default="../datasets/unlabeled_faces",
                help="Path to faces output")

args = vars(ap.parse_args())

# Detector = mtcnn_detector
detector = MTCNN()
# initialize video stream
rtsp = 'rtsp://admin:**graphics@192.168.0.3:554/cam/realmonitor?channel1&subtype=1'
cap = cv2.VideoCapture(rtsp)

# Setup some useful var
faces = 0
frames = 0
max_faces = int(args["faces"])
max_bbox = np.zeros(4)

prevTime = 0
prvFrameTime = 0
while faces < max_faces:
    ret = cap.grab()
    frames += 1

    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    print("calc fps {0}".format(fps))
    if frames % 5 ==0:
        ret, frame = cap.retrieve()
        frameTime = time.time()
        secF = frameTime - prvFrameTime
        prvFrameTime = frameTime
        fpsF = 1/(secF)
        print("calc fps Frame {0}".format(fpsF))
        if not ret: break
        
        # Get all faces on current frame
        reco_tick = time.time()
        frame = cv2.resize(frame, (640, 480))
        reco_tock = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_tick = time.time()
        bboxes = detector.detect_faces(frame)
        detect_tock = time.time()
        if len(bboxes) != 0:
            # Get only the biggest face
            max_area = 0
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                keypoints = bboxe["keypoints"]
                area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                if area > max_area:
                    max_bbox = bbox
                    landmarks = keypoints
                    max_area = area

            max_bbox = max_bbox[0:4]

            # get each of 3 frames
            if frames%3 == 0:
                # convert to face_preprocess.preprocess input
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                     landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
                if not(os.path.exists(args["output"])):
                    os.makedirs(args["output"])
                cv2.imwrite(os.path.join(args["output"], "{}.jpg".format(faces+1)), nimg)
                cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                print("[INFO] {} faces detected".format(faces+1))
                faces += 1
        
        cv2.imshow("Face detection", frame)
        #print("Faces detection time: {}s".format(detect_tock-detect_tick))
        #print("Faces recognition time: {}s".format(reco_tock-reco_tick))
        fpsv = cap.get(cv2.CAP_PROP_FPS)
        print("opencv fps {0}".format(fpsv))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
