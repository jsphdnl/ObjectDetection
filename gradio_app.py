from pathlib import Path
from time import time

import torch
import gradio as gr
from glob import glob

import sys
sys.path.insert(0, './WebsiteYOLO')

from darknet import Darknet
from utils import check_if_file_exists_else_download, predict_and_save, scale_numbers

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class App:

    def __init__(self,
            weights_path='./weights/yolov3.weights',
            config_path='./cfg/yolov3_608x608.cfg',
            labels_path='./data/coco.names',
            font_path='./data/FreeSansBold.ttf',
            examples_glob="./data/*.jpg",
            max_side_size=1280,
            **gr_interface_kwargs,
        ) -> None:
        self.device = torch.device('cpu')
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)  # yolov3_416x416.cfg also available
        self.labels_path = Path(labels_path)
        self.font_path = Path(font_path)
        self.examples = sorted(glob(examples_glob), reverse=True)
        self.max_side_size = max_side_size

        logging.info('Initializing the model...')
        self.model = Darknet(self.config_path)
        logging.info('Loading weights...')
        self.model.load_weights(check_if_file_exists_else_download(self.weights_path))
        self.model.eval()

        self.iface = gr.Interface(
            fn=self.predict,
            inputs=gr.Image(type='pil'),
            outputs=[
                gr.Image(type='pil', label='Image with detected objects'),
                gr.Markdown()
            ],
            examples=self.examples,
            cache_examples=False,
            title='Object Detector',
            description=self.get_desc(),
            article=self.get_article(),
            allow_flagging='never',
            theme=gr.themes.Soft(),
            **gr_interface_kwargs,
        )
        logging.info('Launching Gradio interface...')
        self.iface.launch()

    def predict(self, source_img):
        start_timer = time()
        if source_img is None:
            logging.info('No image provided. Returning None.')
            return None, None
        orig_size = source_img.size
        source_img = self.rescale_img(source_img)
        # inference
        with torch.no_grad():
            predictions, img = predict_and_save(
                source_img, self.model, self.device, self.labels_path, self.font_path,
                orientation=None, save=False
            )
        logging.info(f'Input image dims: {orig_size}. Inference took {time() - start_timer:.2f} sec')
        return img, predictions

    def rescale_img(self, img):
        '''img is a PIL image'''
        W, H = img.size
        H_new, W_new, scale = scale_numbers(H, W, self.max_side_size)
        img = img.resize((W_new, H_new))
        return img

    def get_desc(self):
        return 'Object Detection ' 
    

    def get_article(self):
        return 'More info:\n' 


if __name__ == '__main__':
    App()
