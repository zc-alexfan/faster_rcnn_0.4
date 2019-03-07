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
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.faster_rcnn.vgg16 import vgg16
from easydict import EasyDict as edict
from model.utils.config import cfg

class vgg_extractor(vgg16):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    vgg16.__init__(self, classes, pretrained, class_agnostic)

  @staticmethod
  def _detach2numpy(_tensor):
      return _tensor.cpu().detach().numpy()

  def forward(self, im_data, im_info, gt_boxes, num_boxes):
    batch_size = im_data.size(0)
    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    assert batch_size == 1, "Expecting batch size = 1"
    assert cfg.POOLING_MODE == 'align', "Only consided align when building this. Might need to modify the code a bit"


    image_summary = edict()
    image_summary.gt = edict()
    image_summary.pred = edict()
    image_summary.info = edict()

    image_summary.info.dim_scale = vgg_extractor._detach2numpy(im_info)


    # feed image data to base model to obtain base feature map
    base_feat = self.RCNN_base(im_data)

    # feed base feature map tp RPN to obtain rois
    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
    image_summary.pred.base_feat = vgg_extractor._detach2numpy(base_feat)
    image_summary.pred.rois = vgg_extractor._detach2numpy(rois)

    # default values in 'test' mode
    rois_label = None
    rois_target = None
    rois_inside_ws = None
    rois_outside_ws = None

    rois = Variable(rois)
    """
    (Pdb) base_feat.size()
    torch.Size([15, 512, 37, 56])
    (Pdb) rois.size()
    torch.Size([15, 256, 5])
    (Pdb) batch_size
    15
    15*256=3840
    (Pdb) pooled_feat.size()
    torch.Size([3840, 512, 7, 7]) 

    # all rois feature in the batch
    pooled_feat size for each roi is 512x7x7
    """
    # do roi pooling based on predicted rois
    pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))

    image_summary.pred.pooled_feat = vgg_extractor._detach2numpy(pooled_feat)
    image_summary.info.formatted = False

    # feed pooled features to top model
    pooled_feat = self._head_to_tail(pooled_feat)

    # compute bbox offset
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)

    # compute object classification probability
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score, 1)

    rpn_loss_cls = 0
    rpn_loss_bbox = 0
    RCNN_loss_cls = 0
    RCNN_loss_bbox = 0

    cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
    bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

    #image_summary.pred.bbox_pred = vgg_extractor._detach2numpy(bbox_pred)
    #image_summary.pred.cls_prob = vgg_extractor._detach2numpy(cls_prob)
    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, image_summary


