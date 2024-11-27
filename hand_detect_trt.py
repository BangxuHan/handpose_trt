import random
import time

# import utils.inference as inference_utils  # TRT/TF inference wrappers
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
# import torch
# import torchvision

from logger import logger as log
import threading
_hand_detect_lock = threading.Lock()


def process_data(img, img_size=640):  # 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RG25
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 55, 90], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite('img.jpg', img)


def box_iou(box1, box2):
    def box_area(boxes):
        # box = 4xn
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = np.copy(x)  # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img_size, coords, img0_shape):  # image size 转为 原图尺寸
    # Rescale x1, y1, x2, y2 from 416 to image size
    # print('coords     : ',coords)
    # print('img0_shape : ',img0_shape)
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    # print('gain       : ',gain)
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    # print('pad_xpad_y : ',pad_x,pad_y)
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    # coords[:, :4] = torch.clamp(coords[:, :4], min=0)  # 夹紧区间最小值不为负数
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=None)  # 夹紧区间最小值不为负数
    return coords


def letterbox(img, height=640, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def numpy_nms(boxes, scores, iou_threshold: float):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)
    return keep


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, fast=False, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    fast |= conf_thres > 0.001  # fast mode
    if fast:
        merge = False
        multi_label = False
    else:
        merge = True  # merge for best mAP (adds 0.5ms/img)
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # if multi_label:
        #     i, j = (x[:, 5:] > conf_thres).nonzero().t()
        #     x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        # else:  # best class only

        # conf, j = x[:, 5:].max(1, keepdim=True)
        # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        conf = x[:, 5:]
        j = np.where(x[:, 5:] > conf_thres, 0, 1)
        x = np.concatenate((box, conf, j), 1)

        # # Filter by class
        # if classes:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        i = numpy_nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #         iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #         weights = iou * scores[None]  # box weights
        #         x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #         if redundant:
        #             i = i[iou.sum(1) > 1]  # require redundancy
        #     except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
        #         print(x, i, x.shape, i.shape)
        #         pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


class yolov5_hand_model(object):
    def __init__(self, engine_path, max_batch=1):
        self.img_size = 640
        self.classes = ["Hand"]
        self.num_classes = len(self.classes)
        self.conf_thres = 0.31
        self.nms_thres = 0.45

        # Load TRT engine
        self.det_image_shape = [3, self.img_size, self.img_size]
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
                "Yolov5 hand det trt engine init fail: " + repr(e) + " engine path: " + engine_path)
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

    def predict(self, img_):
        with _hand_detect_lock:
            img = process_data(img_, self.img_size)
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            # img = torch.from_numpy(img).unsqueeze(0)

            pred = self.infer(img)  # 图片检测
            # print(pred)
            # pred = torch.from_numpy(pred)

            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]  # nms

            if detections is None:
                return []

            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img_.shape).round()

            output_dict_ = []
            for *xyxy, conf, cls in detections:
                label = '%s %.2f' % (self.classes[0], conf)
                x1, y1, x2, y2 = xyxy
                output_dict_.append((float(x1), float(y1), float(x2), float(y2), float(conf.item())))
                plot_one_box(xyxy, img_, label=label, color=(0, 175, 255), line_thickness=2)
            return output_dict_


if __name__ == '__main__':
    enginefile = 'onnx_model/yolov5s_1_3_640_640.engine'
    a = yolov5_hand_model(enginefile)
    img = 'sample.jpg'
    img = cv2.imread(img)
    print(a.predict(img))
