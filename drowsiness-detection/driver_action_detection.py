import shutil
from tensorflow import keras
import argparse
import imutils
import time
import cv2
import os
import numpy as np

# USE GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MODEL_PATH = '/Users/phatnguyen/workspace/eroad_driver_detection_resnet_20_epoch'
# shutil.copyfile(DRIVE_PATH + 'eroad_driver_detection_resnet_20_epoch.h5', 'eroad_driver_detection_resnet_20_epoch.h5')
model = keras.models.load_model('eroad_driver_detection_resnet_20_epoch.h5')
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0,
#                 help="index of webcam on system")
# args = vars(ap.parse_args())

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
label_map = {
    0: 'normal',
    1: 'drinking',
    2: 'phone_left',
    3: 'phone_right'
}

time.sleep(1.0)


def get_center_roi(frame):
    # get the Region of Interest (y,y)
    height, width, channel = frame.shape
    midpoint_x = int(width / 2.0)
    midpoint_y = int(height / 2.0)
    left_x = midpoint_x - midpoint_y
    cropped = frame[0:height, left_x:width - left_x]
    return cropped


def predict_action(frameCloned):
    frameCloned = get_center_roi(frameCloned)
    frameCloned = imutils.resize(frameCloned, width=224)

    # uncomment for debug purpose
    # cv2.imshow("predict_action_frame", frame)

    x_predict = []
    x_predict.append(frameCloned)
    x_predict = np.array(x_predict, dtype=np.uint8)
    prediction = model.predict(x_predict)

    label = np.argmax(prediction)

    print(prediction)
    print("Classified: %s" % (label_map.get(label)))


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    rect, frame = vs.read()
    predict_action(frame.copy())

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
