import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
# import torch
import torchvision.transforms as transforms

from logger import logger as log

tags = {'0': 'one', '1': 'five', '2': 'fist', '3': 'ok', '4': 'heart',
        '5': 'yeah', '6': 'three', '7': 'four', '8': 'six',
        '9': 'love', '10': 'gun', '11': 'thumbUp', '12': 'nine', '13': 'pink'}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


def draw_mask_handpose(img_, hand_, x, y, thick=3):
    # thick = 2
    colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    #
    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['1']['x'] + x), int(hand_['1']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x'] + x), int(hand_['1']['y'] + y)),
             (int(hand_['2']['x'] + x), int(hand_['2']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x'] + x), int(hand_['2']['y'] + y)),
             (int(hand_['3']['x'] + x), int(hand_['3']['y'] + y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x'] + x), int(hand_['3']['y'] + y)),
             (int(hand_['4']['x'] + x), int(hand_['4']['y'] + y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['5']['x'] + x), int(hand_['5']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x'] + x), int(hand_['5']['y'] + y)),
             (int(hand_['6']['x'] + x), int(hand_['6']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x'] + x), int(hand_['6']['y'] + y)),
             (int(hand_['7']['x'] + x), int(hand_['7']['y'] + y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x'] + x), int(hand_['7']['y'] + y)),
             (int(hand_['8']['x'] + x), int(hand_['8']['y'] + y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['9']['x'] + x), int(hand_['9']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x'] + x), int(hand_['9']['y'] + y)),
             (int(hand_['10']['x'] + x), int(hand_['10']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x'] + x), int(hand_['10']['y'] + y)),
             (int(hand_['11']['x'] + x), int(hand_['11']['y'] + y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x'] + x), int(hand_['11']['y'] + y)),
             (int(hand_['12']['x'] + x), int(hand_['12']['y'] + y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['13']['x'] + x), int(hand_['13']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x'] + x), int(hand_['13']['y'] + y)),
             (int(hand_['14']['x'] + x), int(hand_['14']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x'] + x), int(hand_['14']['y'] + y)),
             (int(hand_['15']['x'] + x), int(hand_['15']['y'] + y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x'] + x), int(hand_['15']['y'] + y)),
             (int(hand_['16']['x'] + x), int(hand_['16']['y'] + y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x'] + x), int(hand_['0']['y'] + y)),
             (int(hand_['17']['x'] + x), int(hand_['17']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x'] + x), int(hand_['17']['y'] + y)),
             (int(hand_['18']['x'] + x), int(hand_['18']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x'] + x), int(hand_['18']['y'] + y)),
             (int(hand_['19']['x'] + x), int(hand_['19']['y'] + y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x'] + x), int(hand_['19']['y'] + y)),
             (int(hand_['20']['x'] + x), int(hand_['20']['y'] + y)), colors[4], thick)


class gesture_model(object):
    def __init__(self, engine_path, max_batch=1):
        self.img_size = 128

        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_context = self.device.make_context()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.cuda_context.push()
        self.max_batch = max_batch
        if max_batch > self.max_batch:
            self.max_batch = max_batch
        try:
            self.stream = cuda.Stream()
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            assert self.engine
            assert self.context

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                if shape[0] < 0:
                    shape[0] = self.max_batch

                # if shape[2] < 0:
                #     shape[2] = self.det_image_shape[1]
                # if shape[3] < 0:
                #     shape[3] = self.det_image_shape[2]

                np_type = np.float32
                if dtype.name == "BOOL":
                    np_type = np.float32
                elif dtype.name == "HALF":
                    np_type = np.float16
                elif dtype.name == "INT32":
                    np_type = np.int32
                elif dtype.name == "INT8":
                    np_type = np.int32
                # size = np.dtype(trt.nptype(dtype)).itemsize
                size = dtype.itemsize
                for s in shape:
                    size *= s
                allocation = cuda.mem_alloc(size)
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np_type,
                    'shape': list(shape),
                    'allocation': allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0
        except Exception as e:
            log.get().error(
                "Hand gesture classify trt engine init fail: " + repr(e) + " engine path: " + engine_path)
            pass
        finally:
            self.cuda_context.pop()

    def input_spec_max(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec_max(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """

        # Prepare the output data
        out_shape = self.output_spec_max()
        out_shape[0][0] = batch.shape[0]
        output = np.zeros(*out_shape)
        self.cuda_context.push()
        try:
            self.context.set_binding_shape(0, batch.shape)
            # Process I/O and execute the network
            cuda.memcpy_htod_async(self.inputs[0]['allocation'], np.ascontiguousarray(batch), self.stream)
            self.context.execute_async_v2(self.allocations, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(output, self.outputs[0]['allocation'], self.stream)

        except Exception as e:
            log.get().error(
                "classify infer fail: " + repr(e) + " in shape: " + str(batch.shape) + "out put:" + str(out_shape))
            pass
        finally:
            self.cuda_context.pop()

        return output

    def __del__(self):
        if self.cuda_context:
            self.cuda_context.pop()

    def pred_gesture(self, box, pts_, algo_img):
        x1, y1, x2, y2 = box
        pts_hand = {}
        for ptk in range(int(pts_.shape[0] / 2)):
            xh = (pts_[ptk * 2 + 0] * float(x2 - x1))
            yh = (pts_[ptk * 2 + 1] * float(y2 - y1))
            pts_hand[str(ptk)] = {
                "x": xh,
                "y": yh,
            }

        img_mask = np.ones(algo_img.shape, dtype=np.uint8)
        img_mask[:, :, 0] = 255
        img_mask[:, :, 1] = 255
        img_mask[:, :, 2] = 255

        draw_mask_handpose(img_mask, pts_hand, x1, y1, int(((x2 - x1) + (y2 - y1)) / 128))
        # cv2.imshow('img', img_mask)
        # cv2.waitKey(0)

        # 检测手势动作
        s_img_mask = img_mask[y1:y2, x1:x2, :]
        s_img_mask = cv2.resize(s_img_mask, (self.img_size, self.img_size))

        # s_img_mask2 = Image.fromarray(s_img_mask)
        # s_img_mask2 = transform(s_img_mask2)
        # s_img_mask2 = s_img_mask2.unsqueeze(dim=0)
        # s_img_mask2 = s_img_mask2.numpy()

        s_img_mask1 = np.array(s_img_mask) / 255.
        s_img_mask1 = (s_img_mask1 - mean) / std
        s_img_mask1 = s_img_mask1.transpose(2, 0, 1)
        s_img_mask1 = np.expand_dims(s_img_mask1, axis=0)
        s_img_mask1 = s_img_mask1.astype(np.float32)

        # a = s_img_mask2 - s_img_mask1
        # print(a)
        output = self.infer(s_img_mask1)

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=1)

        # output = torch.from_numpy(output)
        # output = torch.softmax(output, 1)
        # pre_tag = torch.argmax(output, dim=1)[0].tolist()

        output = softmax(output)
        pre_tag = np.argmax(output, axis=1)[0]
        # print(output)

        score = output[0][pre_tag]
        gesture_name = tags[str(pre_tag)]
        return gesture_name, score


if __name__ == '__main__':
    enginefile = 'onnx_model/resnet50_1_3_256_256.engine'
    a = gesture_model(enginefile)
    img = 'sample.jpg'
    img = cv2.imread(img)

    pts = np.ndarray([0.61966, 0.90313, 0.40981, 0.72019, 0.3208, 0.52945, 0.35156, 0.37392, 0.46856,
                      0.28354, 0.41048, 0.2715, 0.42059, 0.075387, 0.43822, 0.14168, 0.42653, 0.2864,
                      0.57935, 0.24476, 0.59889, 0.05988, 0.5846, 0.17069, 0.54241, 0.34576, 0.73146,
                      0.26997, 0.7687, 0.096452, 0.7392, 0.1872, 0.68135, 0.33238, 0.87757, 0.33443,
                      0.91588, 0.20791, 0.89052, 0.23191, 0.83295, 0.30845])

    print(a.pred_gesture([1146, 761, 1279, 922], pts, img))
