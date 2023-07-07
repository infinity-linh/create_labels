from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
import os, requests, torch, math, cv2
from typing import List, Optional
import numpy as np
import PIL


class Detection():
    def __init__(self) -> None:     
        self.device:str = "cpu"
        self.half:bool = False
        self.hide_labels: bool = False
        self.hide_conf: bool = False 

        self.img_size:int = 160

        self.conf_thres: float =.25 
        self.iou_thres: float =.45 
        self.max_det:int =  1000
        self.agnostic_nms: bool = False 

        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        self.model = DetectBackend(f"D:/User/Bot_C/ESP32_DOIT/scripts/model_control_arm.pt", device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml("D:/User/Bot_C/ESP32_DOIT/scripts/detection/data/coco.yaml")['names']

        print(self.device)


    def process_image(self, img_src, img_size, stride, half):
        image = letterbox(img_src, img_size, stride=self.stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    def detector_image(self, image_input):
        boxes = []
        clas = []
        img, img_src = self.process_image(image_input, self.img_size, self.stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = self.model(img)
        classes:Optional[List[int]] = None # the classes to keep
        det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, classes, self.agnostic_nms, max_det=self.max_det)[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()
        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)
            
            if class_num == 0 and conf >= 0.5:
                boxes.append(xyxy)
                clas.append(class_num)
                label = None if self.hide_labels else (self.class_names[class_num] if self.hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
        # PIL.Image.fromarray(img_ori)
        return clas, np.array(boxes, dtype=int), img_ori

def center_box(box, image):
    w_image, h_image = image.shape[1], image.shape[0]
    w, h = box[2]-box[0], box[3]-box[1]
    x, y = (box[0]+w//2)/w_image, (box[1]+h//2)/h_image
    return x,y,w/w_image, h/h_image
# de = Detection()
# image = cv2.imread("D:/User/data/image/Val/image_12.png")
# image = cv2.resize(image, (400, 400))
# box, image = de.detector_image(image)

# # box = np.array(box[0], dtype=int)
# print(box)
# cv2.imshow("test", image)
# cv2.waitKey(0)