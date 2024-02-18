import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm

import json

from datasets.arrow_dataset import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            # return img


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


def opencv_loader(path):
    import cv2
    return cv2.imread(path)

# class ImageLabelFolder(Dataset):
class ImageLabelFolder(data.Dataset):
    class imageLabel:
        def __init__(self, image_path, annotate_path):
            self.image = image_path
            try:
                self.annotate = json.load(open(annotate_path))
                if os.path.exists(image_path):
                    self.success = True
            except:
                self.success = False

    def __init__(self, proto, img_root, annotate_root,
                 loader=default_loader):
        protoFile = open(proto, encoding="utf8", errors='ignore')
        content = protoFile.readlines()
        self.datalist = []
        self.loader = loader

        for line in tqdm(content):
            line = line.strip()

            img_path = line.replace('_emotion.json', '.jpg')
            img_path = os.path.join(img_root, img_path)
            annotate_path = os.path.join(annotate_root, line)

            data = self.imageLabel(img_path, annotate_path)
            if data.success:
                self.datalist.append(self.imageLabel(img_path, annotate_path))

        print('data size is ', self.datalist.__len__(), content.__len__())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        cur_imageLabel = self.datalist[index]

        img = self.loader(cur_imageLabel.image)
        labels = cur_imageLabel.annotate


        return {"image": img, "annotate":labels}

    def __len__(self):
        return len(self.datalist)

    def with_transform(self, transformers):
        # Without Useage
        self.transformers = transformers