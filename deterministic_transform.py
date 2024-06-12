from pytorchcv.model_provider import get_model

import torchvision.transforms.v2.functional as TF
from torchvision.transforms import v2

from PIL import Image

from tqdm import tqdm

import torch

from datasets.CUB200 import CUB200
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
rng = np.random.default_rng()

def transform_post(image):
    image = TF.to_dtype(image, dtype=torch.float32, scale=True)
    return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def transform_brightness():
    temp = 1 + rng.random()
    temp = temp if rng.choice([True, False]) else 1/temp
    return lambda x: TF.adjust_brightness(x, temp), temp

def transform_contrast():
    temp = 1 + rng.random()
    temp = temp if rng.choice([True, False]) else 1/temp
    return lambda x: TF.adjust_contrast(x, temp), temp

def transform_gamma():
    temp = 1 + rng.random()
    temp = temp if rng.choice([True, False]) else 1/temp
    return lambda x: TF.adjust_gamma(x, temp), temp

def transform_saturation():
    temp = 1 + rng.random()
    temp = temp if rng.choice([True, False]) else 1/temp
    return lambda x: TF.adjust_saturation(x, temp), temp   

def transform_hue():
    temp = rng.random() - .5
    return lambda x: TF.adjust_hue(x, temp), temp

def transform_blur():
    temp = int(rng.integers(0,5))
    if temp == 0:
      return lambda x: x, temp
    else:
      return lambda x: TF.gaussian_blur(x, kernel_size=(temp*2+1)), temp

def transform_rotate():
    temp = int(rng.integers(-15,15))
    return lambda x: TF.rotate(x, angle=temp), temp

class DeterministicTransform():

  def __init__(self, transform_name='hue', num_transforms = 10):
    self.num_transforms = num_transforms
    self.transform_name = transform_name   
 
  def _get_current(self):
    match self.transform_name:
      case 'brightness':
        return transform_brightness
      case 'contrast':
        return transform_contrast
      case 'gamma':
        return transform_gamma
      case 'saturation':
        return transform_saturation
      case 'hue':
        return transform_hue
      case 'blur':
        return transform_blur
      case 'rotate':
        return transform_rotate
      case _:
        return lambda x:x

  def predict(self, model, dataloader, num_examples=100):
    current_transform = self._get_current()
    model.eval()

    i_acc = []
    j_acc = []
    same_acc = []
    param_acc = []

    with torch.no_grad():
      for i in tqdm(range(num_examples)):
        sample = next(iter(dataloader))        
        image = transform_post(sample[0])
        id = sample[1].item()
        pred = np.argmax(model(image))
        for j in range(self.num_transforms):
          t,p = current_transform()
          image = transform_post(t(sample[0]))
          new_pred = np.argmax(model(image))

          i_acc.append(i)
          j_acc.append(j)
          same_acc.append( pred.item() == new_pred.item())
          param_acc.append(p)

    return pd.DataFrame({'i':i_acc,'j':j_acc,'same':same_acc,'p':param_acc})