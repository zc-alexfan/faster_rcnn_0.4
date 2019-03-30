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
import numpy as np
from tqdm import tqdm
import torch
import pickle
from model.utils.config import cfg, cfg_from_file, cfg_from_list 
from model.faster_rcnn.rois_extractor import rois_extractor
from easydict import EasyDict as edict
import glob

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

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


class DecoupledFasterRCNN():
  def __init__(self, load_name):
    self._module_dir = os.path.dirname(os.path.abspath(__file__))

    self._device = torch.device('cuda:0')

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
    args.load_dir = 'models'
    args.class_agnostic = False
    args.cfg_file = os.path.join(self._module_dir, 'cfgs/vgg16.yml')
    args.mGPUs = False
    args.parallel_type = 0
    args.set_cfgs = None

    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    args.cfg_file = "cfgs/vgg16.yml"
    args.cfg_file = os.path.join(self._module_dir, args.cfg_file)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    cfg.TRAIN.USE_FLIPPED = False

    assert os.path.isfile(load_name), ' cannot find the file %s'%(load_name)

    # initilize the network here.
    fasterRCNN = rois_extractor(class_labels, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    self._args = args
    self._cfg = cfg
    self._fasterRCNN = fasterRCNN

  def predict(self, images):
    pass

#   def rois_predict(self, images, rois):
  def rois_predict(self, dataset, rois_vec, _image_index): 
    fasterRCNN = self._fasterRCNN

    image_path = '/home/alex/vg_data/vg_split/%s/' %('vg_alldata_minival')
    feature_path = os.path.join(image_path, 'features')
    if not os.path.exists(feature_path):
      os.makedirs(feature_path)

    num_images = len(dataset)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1).to(self._device)

    cfg.CUDA = True

    summary_vec = []
    with torch.no_grad():  
      fasterRCNN.to(self._device)
      fasterRCNN.eval()
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0, pin_memory=True)
      data_iter = iter(dataloader)

      for i in tqdm(range(num_images)):
          
          data = next(data_iter)

          im_data.data.resize_(data[0].size()).copy_(data[0])
          scale = data[1].item()
          im_info = torch.FloatTensor([[im_data.size(2), im_data.size(3), scale]]).to(self._device)
          rois = rois_vec[i]

          if rois.shape[0] == 0: # no roi for this image
              continue

          rois = torch.FloatTensor(rois*scale)
          rois = torch.unsqueeze(rois, 0).to(self._device)

          # NOTE: rois should be under the scale of 600, not the original scale
          image_summary = fasterRCNN(im_data, im_info, rois)
          if(i % 1000 == 0):
              print("Cleaning CUDA cache")
              torch.cuda.empty_cache()

          #summary_vec.append(image_summary)
          dump_summary(feature_path, _image_index[i], image_summary, False)
    #return summary_vec


