import base64
import json
import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api

import threading
import argparse
app = Flask("HandposeRecog")
api = Api(app)


_infer_lock = threading.Lock()

import handpose_master

import sys
sys.path.append('components/hand_detect')


class HandRecognition(object):
    def __init__(self, args):
        self.args = args
        self.model = handpose_master.Handpose_Trt(self.args.detect_model_path,
                                                  self.args.keypoint_model_path,
                                                  self.args.gesture_model_path)

    # def __call__(self, img):
    #     print("start handpose recognition process")
    #     result = self.model.predict(img)
    #     return result

    def predict(self, imgs):
        res = []
        for img in imgs:
            result = self.model.predict(img)
            if result is not None:
                res += result
            else:
                res.append([None])
        return res


class HandArgs():
    detect_model_path = 'onnx_model/yolov5s_1_3_640_640.engine'
    keypoint_model_path = 'onnx_model/resnet50_1_3_256_256.engine'
    gesture_model_path = 'onnx_model/resnet18_1_3_128_128.engine'

    def __init__(self):
        pass


class PoseRecognition(Resource):
    def post(self):
        temp = request.get_data(as_text=True)
        data = json.loads(temp)
        images = data['image']
        imagebuf = []
        for imagestr in images:
            imagedata_base64 = base64.b64decode(imagestr)
            np_arr = np.frombuffer(imagedata_base64, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            imagebuf.append(image)

        with _infer_lock:
            det_res = model.predict(imagebuf)
            print(det_res)

        words_res = []
        nlen = len(det_res)

        for i in range(nlen):
            if det_res[i][0] is not None:
                # print('-----------', det_res[i])
                float_res = []
                for num in det_res[i][0]:
                    float_res.append(float(num))
                # float_res = [float(num) for num in det_res[i][0] if isinstance(num, (int, float))]
                # print(float_res)
                temp = {
                    "hand_box": float_res,
                    "hand_pose": det_res[i][1],
                    "pose_conf": det_res[i][2].__float__()
                }
                words_res.append(temp)
            else:
                nlen = 0

        result = {"result_number": nlen,
                  "result_gesture": words_res
                  }
        return app.response_class(json.dumps(result), mimetype='application/json')


api.add_resource(PoseRecognition, '/poserecog')
args = HandArgs()
model = HandRecognition(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='handpose recognition server port')
    args = parser.parse_args()
    port = args.port
    app.run(host='0.0.0.0', port=port, debug=False)
