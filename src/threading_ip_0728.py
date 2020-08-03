import sys
import os
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from imutils import paths
from openpyxl import Workbook
from threading import Thread
from PyQt5 import QtCore, QtGui
import face_preprocess
import numpy as np
import face_model
import threading
import argparse
import pickle
import time
import dlib
import cv2
import imutils
import queue as Queue

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/stream_test.mp4",
    help='Path to output video')


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=1, type=int, help='gpu id')
ap.add_argument('--det', default=1, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Initialize detector
detector = MTCNN()
def getMaxfacebox(bboxes):
    max_area = 0
    for bboxe in bboxes:
        bbox = bboxe["box"]
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        keypoints = bboxe["keypoints"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > max_area:
            max_bbox = bbox
            landmarks = keypoints
            max_area = area
    return max_bbox[0:4], landmarks, max_area
rtsp1 = 'rtsp://admin:**graphics@192.168.0.9:554/cam/realmonitor?channel1&subtype=1'
rtsp2 = 'rtsp://admin:**graphics@192.168.0.16:554/cam/realmonitor?channel1&subtype=1'
rtsp3 = 'rtsp://admin:**graphics@192.168.0.12:554/cam/realmonitor?channel1&subtype=1'
# rtsp3 = 'rtsp://admin:**GRAPHICS@163.152.162.198:554/cam/realmonitor?channel1&subtype=1'

frame_width = 640
frame_height = 480
save_width = frame_width
save_height = int(save_width/frame_width*frame_height)
bboxes = []
gframe1 = Queue.Queue()
gframe2 = Queue.Queue()
gframe3 = Queue.Queue()
gframe =  Queue.Queue()


thread_lock1 = threading.Lock()
thread_lock2 = threading.Lock()
thread_lock3 = threading.Lock()

class VideoStreamWidget(object):
    def __init__(self, link, camname, src=0):
        self.capture = cv2.VideoCapture(link)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.link = link
        self.frames = 0
        #self.gframe = q
        print(camname)
        print(link)

    def update(self):
        # Read the next frame from the stream in a different thread
        global gframe
        while True:
            if self.capture.isOpened():
                
                self.status = self.capture.grab()        #output camera
                self.frames += 1
                (self.status, self.frame) = self.capture.retrieve()
                             
                if not self.status :
                    print(self.camname+" : %d"%self.status)
                    self.capture = cv2.VideoCapture(self.link)
                    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
                    self.capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
                    continue
                if self.frames % 8 == 0:
                    frame = cv2.resize(self.frame, (save_width, save_height))
                    gframe.put(frame)
                    print("stream"+ self.camname)
                    
            time.sleep(.01)
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.frame, width=400)
        cv2.imshow('Frame ' + self.camname, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
class VideoStreamWidget2(object):
    def __init__(self, link, camname, src=0):
        self.capture = cv2.VideoCapture(link)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.link = link
        self.frames = 0
        #self.gframe = q
        print(camname)
        print(link)

    def update(self):
        # Read the next frame from the stream in a different thread
        global gframe2
        while True:
            if self.capture.isOpened():
                self.status = self.capture.grab()        #output camera
                self.frames += 1
                
                (self.status, self.frame) = self.capture.retrieve()
                
              
                if not self.status :
                    print(self.camname+" : %d"%self.status)
                    self.capture = cv2.VideoCapture(self.link)
                    continue
                if self.frames % 8 == 0:
                    frame = cv2.resize(self.frame, (save_width, save_height))
                    gframe2.put(frame)
                    print("stream"+ self.camname)
                    
            time.sleep(.01)
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.frame, width=400)
        cv2.imshow('Frame ' + self.camname, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
class VideoStreamWidget3(object):
    def __init__(self, link, camname, src=0):
        self.capture = cv2.VideoCapture(link)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, False)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.link = link

        self.frames = 0
        #self.gframe = q
        print(camname)
        print(link)

    def update(self):
        # Read the next frame from the stream in a different thread
        global gframe3
        while True:
            if self.capture.isOpened():
                self.status = self.capture.grab()        #output camera
                self.frames += 1
                (self.status, self.frame) = self.capture.retrieve()
                
                if not self.status :
                    print(self.camname+" : %d"%self.status)
                    self.capture = cv2.VideoCapture(self.link)
                    continue
                if self.frames % 8 == 0:
                    frame = cv2.resize(self.frame, (save_width, save_height))
                    gframe3.put(frame)
                    print("stream"+ self.camname)
                    
            time.sleep(.01)
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.frame, width=400)
        cv2.imshow('Frame ' + self.camname, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
            
class VideoProcess(object):
    def __init__(self, camname, thread_lock):
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.thread_lock = thread_lock
        #self.gframe = q
    def run(self):
        global gframe
        while True:
            if(not gframe.empty()):
                self.Currframe = gframe.get()
                facedetect_tick1 = time.time()
                #self.thread_lock.acquire()
                #bboxes = detector.detect_faces(self.Currframe)
                #self.thread_lock.release()
                facedetect_tock1 = time.time()
                self.frame = self.Currframe
                
                if len(bboxes) != 0:
                    print("detect3"+ self.camname)
                    print("Faces recognition time: {}s".format(facedetect_tock1-facedetect_tick1))
                    #self.show_frame()
                else:
                    print(self.camname + "No Faces time: {}s".format(facedetect_tock1-facedetect_tick1))
                gframe.task_done()
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.Currframe, width=200)
        cv2.imshow('Frame dectecting' + self.camname, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)


class VideoProcess2(object):
    def __init__(self, camname, thread_lock):
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.thread_lock = thread_lock
        #self.gframe = q
    def run(self):
        global gframe2
        while True:
            if(not gframe2.empty()):
                self.Currframe = gframe2.get()
                facedetect_tick2 = time.time()
                #self.thread_lock.acquire()
                #bboxes = detector.detect_faces(self.Currframe)
                #self.thread_lock.release()
                facedetect_tock2 = time.time()
                self.frame = self.Currframe
                
                if len(bboxes) != 0:
                    print("detect3"+ self.camname)
                    print("Faces recognition time: {}s".format(facedetect_tock2-facedetect_tick2))
                    #self.show_frame()
                else:
                    print(self.camname + "No Faces time: {}s".format(facedetect_tock2-facedetect_tick2))
                gframe2.task_done()
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.Currframe, width=200)
        cv2.imshow('Frame dectecting' + self.camname, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

class VideoProcess3(object):
    def __init__(self, camname, thread_lock):
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        self.camname = camname
        self.thread_lock = thread_lock
        #self.gframe = q
    def run(self):
        global gframe3
        while True:
            if(not gframe3.empty()):
                self.Currframe = gframe3.get()
                facedetect_tick3 = time.time()
                # self.thread_lock.acquire()
                #bboxes = detector.detect_faces(self.Currframe)
                # self.thread_lock.release()
                facedetect_tock3 = time.time()
                self.frame = self.Currframe
                
                if len(bboxes) != 0:
                    print("detect3"+ self.camname)
                    print("Faces recognition time: {}s".format(facedetect_tock3-facedetect_tick3))
                    #self.show_frame()
                else:
                    print(self.camname + "No Faces time: {}s".format(facedetect_tock3-facedetect_tick3))
                gframe3.task_done()
    def show_frame(self):
        # Display frames in main program
        frame = imutils.resize(self.Currframe, width=200)
        cv2.imshow('Frame dectecting' + self.camname, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)        
        
if __name__ == '__main__':
    video_stream_widget1 = VideoStreamWidget(rtsp1,"Cam1")
    video_stream_widget2 = VideoStreamWidget2(rtsp2,"Cam2")
    video_stream_widget3 = VideoStreamWidget3(rtsp3,"Cam3")
    cam1_process = VideoProcess("Cam1", thread_lock1)
    cam2_process = VideoProcess2("Cam2", thread_lock2)
    cam3_process = VideoProcess3("Cam3", thread_lock3)

    while True:
        try:
            video_stream_widget1.show_frame()
            video_stream_widget2.show_frame()
            video_stream_widget3.show_frame()
            cam1_process.show_frame()
            cam2_process.show_frame()
            cam3_process.show_frame()
        except AttributeError:
            pass
