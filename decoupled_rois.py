# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
from tqdm import tqdm
import torch
import pickle
from model.utils.config import cfg, cfg_from_file, cfg_from_list 
from model.faster_rcnn.rois_extractor import rois_extractor
from easydict import EasyDict as edict

import glob
from scipy.misc import imread
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

from torch.utils.data import Dataset
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
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                        cfg.TRAIN.MAX_SIZE)

    data = torch.from_numpy(im)
    data_height, data_width = data.size(0), data.size(1)
    data = data.permute(2, 0, 1)

    return (data, im_scale)

  def __len__(self):
    return len(self._image_urls)

def dump_summary(feature_path, image_jpg_id, im_summary, isUnion):
    # packing
    out = {}
    prefix = ""
    if isUnion:
        prefix = 'union_'

    out['%spred_pooled_feat' %(prefix)] = im_summary.pred.pooled_feat.reshape(-1, 4096)
    out['%spred_cls_prob'%(prefix)] = im_summary.pred.cls_prob.reshape(-1, 151)
    out['%sinfo_dim_scale'%(prefix)] = im_summary.info.dim_scale

    curr_im_path = feature_path + "/" + image_jpg_id

    for k in out.keys():
        np.save(curr_im_path + '.' + k, out[k])

if __name__ == '__main__':
  device = torch.device('cuda:0')

  class_labels = ['__background__', 'bush', 'kite', 'laptop', 'bear', 'paper', 'shoe', 'chair', 'ground', 'flowers', 'tire',
     'cup', 'sky', 'bench', 'window', 'bike', 'board', 'hat', 'plate', 'woman', 'handle', 'food', 'trees', 'wave',
     'giraffe', 'background', 'foot', 'shadow', 'clouds', 'button', 'shelf', 'bag', 'sand', 'nose', 'rock', 'sidewalk',
     'glasses', 'fence', 'people', 'house', 'sign', 'hair', 'street', 'zebra', 'mirror', 'logo', 'girl', 'arm', 'flower',
     'leaf', 'clock', 'dirt', 'lights', 'boat', 'bird', 'pants', 'umbrella', 'bed', 'leg', 'reflection', 'water', 'tracks',
     'sink', 'trunk', 'post', 'box', 'boy', 'cow', 'shoes', 'leaves', 'skateboard', 'pillow', 'road', 'letters', 'wall',
     'jeans', 'number', 'pole', 'table', 'writing', 'cloud', 'sheep', 'horse', 'eye', 'top', 'seat', 'tail', 'vehicle', 'brick',
     'legs', 'banana', 'head', 'door', 'shorts', 'bus', 'motorcycle', 'glass', 'flag', 'train', 'child', 'line', 'ear', 'neck',
     'car', 'cap', 'tree', 'roof', 'cat', 'coat', 'grass', 'toilet', 'player', 'airplane', 'glove', 'helmet', 'shirt', 'floor', 'bowl',
     'snow', 'field', 'lamp', 'elephant', 'tile', 'beach', 'pizza', 'wheel', 'picture', 'plant', 'ball', 'spot', 'hand', 'plane', 'mouth',
     'stripe', 'letter', 'vase', 'man', 'building', 'surfboard', 'windows', 'light', 'counter', 'lines', 'dog', 'face', 'jacket',
     'person', 'part', 'truck', 'bottle', 'wing']
  assert len(class_labels) == 151

  args = edict()
  args.isUnion = False
  args.dataset = 'vg'
  args.net = 'vgg16'
  args.load_dir = 'models'
  args.checksession = 1
  args.checkepoch = 8
  args.checkpoint = 3264
  args.class_agnostic = False
  args.cuda = True
  args.cfg_file = 'cfgs/vgg16.yml'
  args.large_scale = False
  args.mGPUs = False
  args.parallel_type = 0
  args.set_cfgs = None
  args.vis = False
  



  num_classes = len(class_labels)
  isUnion = args.isUnion


  print("Extracting union: %d"%(isUnion))

  datasplit = 'vg_alldata_smallval'
  datasplit = 'vg_alldata_smalltrain'
  datasplit = 'vg_alldata_minival'
  image_path = '/home/alex/vg_data/vg_split/%s/' %(datasplit)
  rois_path = '/home/alex/faster-rcnn.pytorch/data/rois_interface/%s/'%(datasplit)

  image_extension = ".jpg"
  image_index = glob.glob(os.path.join(image_path, "*" + image_extension))
  image_index = [os.path.basename(x)[:-len(image_extension)] for x in image_index]

  feature_path = os.path.join(image_path, 'features')

  if not os.path.exists(feature_path):
    os.makedirs(feature_path)

  dataset = roibatchLoader(image_path, image_index, image_extension)
  num_images = len(dataset)
  max_per_image = 100

  metaInfo = edict()
  
  print('Called with args:')
  print(args)

  np.random.seed(1)
  assert args.dataset == 'vg'
  assert args.net == 'vgg16'

  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)


  import pdb; pdb.set_trace() 

  #print('Using config:')
  #pprint.pprint(cfg)


  cfg.TRAIN.USE_FLIPPED = False
  metaInfo.imdb_image_index = image_index

  meta_file = os.path.join(feature_path, "meta.pkl")
  with open(meta_file, 'wb') as f:
      pickle.dump(metaInfo, f, pickle.HIGHEST_PROTOCOL)


  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  fasterRCNN = rois_extractor(class_labels, pretrained=False, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1).to(device)

  if args.cuda:
    cfg.CUDA = True

  with torch.no_grad():  
    fasterRCNN.to(device)
    fasterRCNN.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                              shuffle=False, num_workers=0, pin_memory=True)
    data_iter = iter(dataloader)

    for i in tqdm(range(num_images)):
        
        data = next(data_iter)

        im_data.data.resize_(data[0].size()).copy_(data[0])
        scale = data[1].item()
        im_info = torch.FloatTensor([[im_data.size(2), im_data.size(3), scale]]).to(device)

        if isUnion:
            # (x1, y1, x2, y2) -> (0, x1, y1, x2, y2)
            rois = np.load(rois_path + str(image_index[i]) + ".union_boxes.npy" )
            left = np.zeros((rois.shape[0], 1))
            rois = np.concatenate((left, rois), axis=1)
        else:
            # (x1, y1, x2, y2, label) -> (0, x1, y1, x2, y2)
            rois = np.load(rois_path + str(image_index[i]) + ".boxes.npy" )
            rois[:, 1:] = rois[:, 0:-1] 
            rois[:, 0] = 0

        if rois.shape[0] == 0:
            continue

        rois = torch.FloatTensor(rois*scale)
        rois = torch.unsqueeze(rois, 0).to(device)

        # NOTE: rois should be under the scale of 600, not the original scale
        image_summary = fasterRCNN(im_data, im_info, rois)
        if(i % 1000 == 0):
            print("Cleaning CUDA cache")
            torch.cuda.empty_cache()

        dump_summary(feature_path, image_index[i], image_summary, isUnion)






