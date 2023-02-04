import argparse
import queue
from this import d
import threading
import signal
from pathlib import Path

import cv2
import depthai
import numpy as np
from imutils.video import FPS
from math import cos, sin

import os
from datetime import timedelta, datetime
import blobconverter

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")
args = parser.parse_args()

# Will get rid of this when cleaning up code. face/gaze uses cam or video file
camera = True

# Database and video_size
VIDEO_SIZE = (1072, 1072)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

# Helper classes for F_ID
class HostSync:
    def __init__(self):
        self.array = []
    def add_msg(self, msg):
        self.array.append(msg)
    def get_msg(self, timestamp):
        def getDiff(msg, timestamp):
            return abs(msg.getTimestamp() - timestamp)
        if len(self.array) == 0: return None

        self.array.sort(key=lambda msg: getDiff(msg, timestamp))

        # Remove all frames that are older than 0.5 sec
        for i in range(len(self.array)):
            j = len(self.array) - 1 - i
            if getDiff(self.array[j], timestamp) > timedelta(milliseconds=500):
                self.array.remove(self.array[j])

        if len(self.array) == 0: return None
        return self.array.pop(0)


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)
    def drawContours(self, frame, points):
        cv2.drawContours(frame, [points], 0, self.bg_color, 6)
        cv2.drawContours(frame, [points], 0, self.color, 2)


class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{databases}/{self.name}", *db_)
        self.adding_new = False


def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()
    pipeline.setOpenVINOVersion(depthai.OpenVINO.VERSION_2021_2)
    openvino_version = '2021.2'

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300, 300)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(depthai.CameraBoardSocket.RGB)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    print("Creating Mono Left & Right plus Stereo...")
    monoLeft = pipeline.create(depthai.node.MonoCamera)
    monoRight = pipeline.create(depthai.node.MonoCamera)
    stereo = pipeline.create(depthai.node.StereoDepth)

    monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)
    monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.initialConfig.setConfidenceThreshold(255)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # Face Detection NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
        version=openvino_version
    ))

    cam.preview.link(face_nn.input)

    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)


    # Object Tracker Neural Network
    print("Creating Object Tracker Neural Network...")
    spatialDetectionNetwork = pipeline.create(depthai.node.MobileNetSpatialDetectionNetwork)
    objectTracker = pipeline.create(depthai.node.ObjectTracker)

    trackerOut = pipeline.create(depthai.node.XLinkOut)
    trackerOut.setStreamName("tracklets")

    spatialDetectionNetwork.setBlobPath(str(Path('models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute()))
    spatialDetectionNetwork.setConfidenceThreshold(0.7)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    cam.preview.link(spatialDetectionNetwork.input)


    objectTracker.setDetectionLabelsToTrack([15])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
    objectTracker.setTrackerType(depthai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    # objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)
    objectTracker.passthroughTrackerFrame.link(cam_xout.input)
    objectTracker.out.link(trackerOut.input)


    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    spatialDetectionNetwork.out.link(objectTracker.inputDetections)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    return pipeline
    


class Main:
    def __init__(self, device):
        self.device = device
        print("Starting pipeline...")
        self.device.startPipeline()

        self.cam_out = self.device.getOutputQueue("cam_out")
       
        self.frame = None
        self.face_box_q = queue.Queue()
        self.bboxes = []
        self.left_bbox = None
        self.right_bbox = None
        self.nose = None
        self.pose = None
        self.gaze = None

        self.running = True
        self.fps = FPS()
        self.fps.start()

    def face_thread(self):
        face_nn = self.device.getOutputQueue("face_nn")
        landmark_in = self.device.getInputQueue("landmark_in")
        pose_in = self.device.getInputQueue("pose_in")

        while self.running:
            if self.frame is None:
                continue
            try:
                bboxes = np.array(face_nn.get().getFirstLayerFp16())
            except RuntimeError as ex:
                continue
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

            for raw_bbox in self.bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                det_frame = self.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                land_data = depthai.NNData()
                land_data.setLayer("0", to_planar(det_frame, (48, 48)))
                landmark_in.send(land_data)

                pose_data = depthai.NNData()
                pose_data.setLayer("data", to_planar(det_frame, (60, 60)))
                pose_in.send(pose_data)

                self.face_box_q.put(bbox)

    