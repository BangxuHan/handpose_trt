import random
import time

# import utils.inference as inference_utils  # TRT/TF inference wrappers
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
# import torch
import torchvision

from logger import logger as log
import threading
_hand_keypoint_lock = threading.Lock()


class handpose_model(object):
    def __init__(self, engine_path, max_batch=1):
        self.img_size = 256
        self.num_classes = 42

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
                "Hand keypoint det trt engine init fail: " + repr(e) + " engine path: " + engine_path)
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
                "det infer fail: " + repr(e) + " in shape: " + str(batch.shape) + "out put:" + str(out_shape))
            pass
        finally:
            self.cuda_context.pop()

        return output

    def __del__(self):
        if self.cuda_context:
            self.cuda_context.pop()

    def predict(self, img):
        with _hand_keypoint_lock:

            if not ((img.shape[0] == self.img_size) and (img.shape[1] == self.img_size)):
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

            img_ = img.astype(np.float32)
            img_ = (img_ - 128.) / 256.

            img_ = img_.transpose(2, 0, 1)
            # img_ = torch.from_numpy(img_)
            img_ = np.expand_dims(img_, axis=0)

            # if self.use_cuda:
            #     img_ = img_.cuda()  # (bs, 3, h, w)

            pre_ = self.infer(img_)
            # output = pre_.cpu().detach().numpy()
            output = np.squeeze(pre_)

            return output


if __name__ == '__main__':
    enginefile = 'onnx_model/squeezenet1_1_3_256_256.engine'
    a = handpose_model(enginefile)
    img = '1706772609350.jpg'
    img = cv2.imread(img)
    print(a.predict(img))
