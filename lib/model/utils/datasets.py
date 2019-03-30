import numpy as np
import cv2
from scipy.misc import imread
from torch.utils.data import Dataset
import os
import torch

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


class roibatchLoader(Dataset):
  def __init__(self, image_path, image_urls, image_extension):
    self._image_urls = image_urls
    self._image_path = image_path
    self._image_extension = image_extension

  def __getitem__(self, index):
    im = imread(os.path.join(\
            self._image_path, self._image_urls[index] + self._image_extension))

    if len(im.shape) == 2:
        imnew = np.zeros((im.shape[0], im.shape[1], 3))
        imnew[:, :, 0] = im 
        imnew[:, :, 1] = im 
        imnew[:, :, 2] = im 
        im = imnew
    else:
        im = im[:, :, ::-1] # rgb -> bgr
    target_size = 600
    #im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    max_size = 1000
    im, im_scale = prep_im_for_blob(im, pixel_means, target_size, max_size)

    data = torch.from_numpy(im)
    data_height, data_width = data.size(0), data.size(1)
    data = data.permute(2, 0, 1)

    return (data, im_scale)

  def __len__(self):
    return len(self._image_urls)


