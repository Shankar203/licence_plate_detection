import argparse
from PIL import Image
import numpy as np
import cv2
import time
# import logging
import pandas as pd
import torch
# from torchvision.utils import save_image
# import os
import gradio as gr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers
transformers.utils.logging.set_verbosity_error()
st = time.time()
# parser = argparse.ArgumentParser(description='License Plate Recognition')
# parser.add_argument("img_path",type=str, help="path of the image")
# parser.add_argument("--save", type=bool, default=True, help="save annotated image")
# parser.add_argument("--output_dir",type=str, default="./results",help="output dir for annotated image/video")
# parser.add_argument("--br_ad",'--brightness_adjustment',type=bool, default=True,help="adjust brightness for low-light images")
# parser.add_argument("--model",type=str,default="microsoft/trocr-small-printed",help="TrOCR model to use")
# args = parser.parse_args()

args = {'model':'microsoft/trocr-base-printed'}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_PATH = 'best.pt'
yolov5 = torch.hub.load('./yolov5','custom',
                        path=CKPT_PATH,
                        source='local',
                        force_reload=True,
                        verbose=False)

model = VisionEncoderDecoderModel.from_pretrained(args['model'],cache_dir="./cache/"+args['model'])
processor = TrOCRProcessor.from_pretrained(args['model'],cache_dir="./cache/"+args['model']) 
model.eval()
model = model.to(device)

def frame_extract(vid_path):
    vidObj = cv2.VideoCapture(vid_path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def detect(img):
    pred = yolov5(img, size=1280, augment=False)
    return pred.pandas().xyxy[0]

def gen_text(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def label_gen(img):
    
    # img = Image.open(img_path).convert('RGB')
    img = np.array(img,dtype=np.float32)
    # if True:
    #     img,_,_ = automatic_brightness_and_contrast(img,5)
    data_img = detect(img)
    # print(data_img,img.shape)
    results = {'labels':[],'confidence':[],'ratio':[]}
    # for i,row in data_img.iterrows():
    #     if row['confidence']>0.5:
    #         xmin,xmax,ymin,ymax = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
    #         # if save:
    #         cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
    #         plate = img[ymin:ymax,xmin:xmax,:]
    #         plate = torch.from_numpy(plate)
    #         # print(plate.shape)
            
    #         w = xmax-xmin
    #         h = ymax-ymin
    #         ratio = w/h
    #         label = gen_text(plate)
    #         # if save:
    #         #     save_image(plate.permute((2,0,1))/255.,out_dir+'/'+label+".png")
    #         results['labels'].append(label)
    #         results['confidence'].append(round(row['confidence'],2))
    #         results['ratio'].append(round(ratio,2))
    # if save:        
    #     im = Image.fromarray(img)
    #     im.save(out_dir+"/image.png")
    return results


def automatic_brightness_and_contrast(image, clip_hist_percent=5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


if __name__=='__main__':
    lpr = gr.Interface(fn=label_gen, inputs="image", outputs=["dataframe","image"])
    lpr.launch(share=True)

