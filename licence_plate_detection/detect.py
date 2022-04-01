import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel

YOLO_CKPT_PATH = "yolov5/weights/best.pt"
TROCR_CKPT_PATH = "trocr-small-printed"
yolov5 = torch.hub.load('yolov5', 'custom',
                        path=YOLO_CKPT_PATH,
                        source='local',
                        force_reload=True,
                        verbose=False)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_CKPT_PATH)
processor = AutoProcessor.from_pretrained(TROCR_CKPT_PATH)


def detect(img):
    pred = yolov5(img, size=1280, augment=False)
    return pred.pandas().xyxy[0]


def gen_text(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def label_gen(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    bboxs = detect(img)
    xmin, xmax, ymin, ymax = int(bboxs.loc[0, 'xmin']), int(bboxs.loc[0, 'xmax']), int(bboxs.loc[0, 'ymin']), int(bboxs.loc[0, 'ymax'])
    img_cropped = img[ymin:ymax, xmin:xmax]
    results = gen_text(img_cropped)
    return str(results)
