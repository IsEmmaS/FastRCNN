import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch_snippets import show
from torchvision.ops import nms

from dataloader import target2label, preprocess_image
from model import get_model

warnings.filterwarnings('ignore')


def decode_outputs(outputs):
    """
    convert tensor outputs to np array
    """

    bbs = outputs['boxes'].detach().cpu().numpy().astype(np.uint16)
    labels = np.array(
        [target2label[i] for i in outputs['labels'].detach().cpu().numpy()]
    )
    confs = outputs['scores'].detach().cpu().numpy()
    ixs = nms(
        torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), .05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]

    return bbs.tolist(), confs.tolist(), labels.tolist()


def get_image(image_root_path, filename):
    """
    load one Image
    :param image_root_path: str path to image
    :param filename: str filename
    :return: image
    """

    image_path = image_root_path + filename
    img = Image.open(image_path).convert('RGB')
    img = np.array(img.resize(size=(224, 224),
                              resample=Image.BILINEAR)) / 255.
    image = preprocess_image(img)
    return image


def get_images_with_prefix(image_root_path, prefix='demo'):
    """
    Return images with given prefix in folder image_root_path as a list.
    """

    image_files = []
    for root, dirs, files in os.walk(image_root_path):
        for file in files:
            if file.startswith(prefix) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(get_image(root, file))

    return image_files


# Eval Model
model = get_model()
model.load_state_dict(torch.load('./weights/model.pth'))
model.eval()

images = get_images_with_prefix('assets/', prefix='demo')

outputs = model(images)
for ix, output in enumerate(outputs):
    bbs, confs, labels = decode_outputs(output)
    info = [f'{label}@{conf:.2f}' for label, conf in zip(labels, confs)]
    print("Number of bounding boxes:", len(bbs))
    show(images[ix].cpu().permute(1, 2, 0), bbs=bbs, texts=labels, sz=3, text_sz=12)
