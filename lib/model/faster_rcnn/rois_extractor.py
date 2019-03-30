# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.faster_rcnn.vgg16 import vgg16
from easydict import EasyDict as edict

class rois_extractor(vgg16):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    vgg16.__init__(self, classes, pretrained, class_agnostic)

  @staticmethod
  def _detach2numpy(_tensor):
      return _tensor.cpu().detach().numpy()

  def forward(self, im_data, im_info, rois):
    batch_size = im_data.size(0)
    #im_info = im_info.data #### scale might be a problem
    assert batch_size == 1, "Expecting batch size = 1"

    image_summary = edict()
    image_summary.pred = edict()
    image_summary.info = edict()

    image_summary.info.dim_scale = rois_extractor._detach2numpy(im_info).squeeze()

    # feed image data to base model to obtain base feature map
    base_feat = self.RCNN_base(im_data)
    image_summary.pred.base_feat = rois_extractor._detach2numpy(base_feat).squeeze()


    # roi align pooling 
    pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))


    # feed pooled features to top model
    pooled_feat = self._head_to_tail(pooled_feat)

    image_summary.pred.pooled_feat = rois_extractor._detach2numpy(pooled_feat)

    # compute object classification probability
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score, 1)

    cls_prob = cls_prob.view(batch_size, rois.size(1), -1)

    image_summary.pred.cls_prob = rois_extractor._detach2numpy(cls_prob)

    return image_summary


