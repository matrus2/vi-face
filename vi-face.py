import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
import datetime
import signal
import sys
import imutils
import pickle
import cv2
import time
import numpy as np
from led import HD44780
from slack import Slack


def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


class GpioHandler(object):
    def __init__(self):
        self.acquire_button = 17
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.ldc_screen = HD44780()
        self.camera = PiCamera()
        self.camera.resolution = (1920, 1080)
        self.rawCapture = PiRGBArray( self.camera)
        time.sleep(2)
        GPIO.setup(self.acquire_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.acquire_button, GPIO.FALLING, callback=self.my_callback, bouncetime=200)

        print("[INFO] loading face detector...")
        self.protoPath = "face_detection_model/deploy.prototxt"
        self.modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

        self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
        self.label_encoder = pickle.loads(open("output/le.pickle", "rb").read())

        self.bot = Slack()

    def __enter__(self):
        # self.camera.start_preview()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera.close()
        GPIO.cleanup()

    def signal_handler(self, sig, frame):
        self.camera.close()
        GPIO.cleanup()

    def my_callback(self, channel):
        print("button:", channel, " pressed in gpioHandler")
        time.sleep(1)
        # self.ldc_screen.message("               \n                ")
        # self.ldc_screen.clear()
        if not GPIO.input(self.acquire_button):
            print("button hold")
            st = str(datetime.datetime.utcnow()).replace('.', '_')
            self.camera.capture(self.rawCapture, format="bgr")
            image = self.rawCapture.array

            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            print(h, w)

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            self.detector.setInput(imageBlob)
            detections = self.detector.forward()

            for i in range(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > 0.8:

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                     (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(faceBlob)
                    vec = self.embedder.forward()

                    preds = self.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = self.label_encoder.classes_[j]
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    print(text)
                    if 'unknown' not in text:
                        self.bot.send_slack_message_sandwich()
                    self.ldc_screen.message("Recognized\n" + text)
                    time.sleep(5)
                    self.ldc_screen.clear()

                self.rawCapture.truncate(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    with GpioHandler() as gpio_handler:
        while True:
            time.sleep(1)
            print("main loop")
