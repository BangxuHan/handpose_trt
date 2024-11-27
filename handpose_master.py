import cv2
import numpy as np
from hand_detect_trt import yolov5_hand_model
from hand_keypoint_trt import handpose_model
from hand_gesture_trt import gesture_model

import threading
import argparse

_infer_lock = threading.Lock()


class Handpose_Trt(object):
    def __init__(self, detect_model_path, keypoint_model_path, gesture_model_path):
        self.hand_detect_model = yolov5_hand_model(detect_model_path)
        self.handpose_model = handpose_model(keypoint_model_path)
        self.gesture_model = gesture_model(gesture_model_path)
        self.drop_score = 0.7

    def predict(self, img):
        algo_img = img.copy()
        print("start handpose recognition process")
        hand_bbox = self.hand_detect_model.predict(img)
        if (hand_bbox is None) or len(hand_bbox) == 0:
            return None
        gesture_list = []
        for h_box in hand_bbox:
            x_min, y_min, x_max, y_max, score = h_box
            w_ = max(abs(x_max - x_min), abs(y_max - y_min))
            if w_ < 60:
                continue
            w_ = w_ * 1.26

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2

            x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

            x1 = np.clip(x1, 0, img.shape[1] - 1)
            x2 = np.clip(x2, 0, img.shape[1] - 1)

            y1 = np.clip(y1, 0, img.shape[0] - 1)
            y2 = np.clip(y2, 0, img.shape[0] - 1)

            gesture_box = [x1, y1, x2, y2]
            # box = [float(x1), float(y1), float(x2), float(y2)]

            pts_ = self.handpose_model.predict(algo_img[y1:y2, x1:x2, :])  # 预测手指关键点

            gesture_name, gesture_score = self.gesture_model.pred_gesture(gesture_box, pts_, algo_img)
            # print(gesture_name)
            # if gesture_score >= self.drop_score:
            #     gesture_list.append([gesture_box, gesture_name, gesture_score])
            gesture_list.append([gesture_box, gesture_name, gesture_score])

        return gesture_list


if __name__ == '__main__':
    detect_enginefile = 'onnx_model/yolov5s_1_3_640_640.engine'
    keypoint_enginefile = 'onnx_model/squeezenet1_1_3_256_256.engine'
    gesture_enginefile = 'onnx_model/resnet18_1_3_128_128.engine'

    hand_master = Handpose_Trt(detect_enginefile, keypoint_enginefile, gesture_enginefile)
    image_path = 'sample.jpg'
    # with open(image_path, "rb") as f:
    #     data = f.read()
    # imagedata_base64 = base64.b64decode(data)
    # np_arr = np.frombuffer(imagedata_base64, dtype=np.uint8)
    img = cv2.imread(image_path)
    out = hand_master.predict(img)
    print(out)
