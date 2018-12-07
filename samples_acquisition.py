import time
import RPi.GPIO as GPIO
import picamera
import datetime
import signal
import sys


def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


class GpioHandler(object):
    def __init__(self):
        self.acquire_button = 17
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.camera = picamera.PiCamera()
        self.camera.resolution = (1024, 768)
        time.sleep(2)
        GPIO.setup(self.acquire_button, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.acquire_button, GPIO.FALLING, callback=self.my_callback, bouncetime=200)

    def __enter__(self):
        self.camera.start_preview()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera.close()
        GPIO.cleanup()

    def signal_handler(self, sig, frame):
        self.camera.close()
        GPIO.cleanup()

    def my_callback(self, channel):
        print("button:", channel, " pressed in gpioHandler")
        time.sleep(1)
        if not GPIO.input(self.acquire_button):
            print("button hold")
            st = datetime.datetime.utcnow()
            self.camera.capture((str(st) + ".png"))
            print("camera output captured")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    with GpioHandler() as gpio_handler:
        while True:
            time.sleep(1)
            print("main loop")
