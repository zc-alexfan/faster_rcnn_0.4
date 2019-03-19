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
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.faster_rcnn.generic_extractor import generic_extractor
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

def formalize_bbox(_im_summary): 
    """
    Extract bboxes from all classes and return a list of bbox. 
    Each element of the list is in the form: [x1, y1, x2, y2, class_id, score]. 
    The returned list is sorted descendingly according to score. 
    """
    boxes = [] # each element: x, y, w, h, class_id, score 
    probs = [] # prob distribution for each bounding box
    feats = [] # pooled features
    
    for class_id, items in enumerate(_im_summary.pred.boxes):
        for bbox in items:
            x1, y1, x2, y2, score = bbox
            boxes.append([x1, y1, x2, y2, class_id, score])
    
    for class_id, items in enumerate(_im_summary.pred.cls_prob):
        for cls_prob in items:                
            probs.append(cls_prob)
    assert len(boxes) == len(probs)

    for class_id, items in enumerate(_im_summary.pred.pooled_feat):
        for f in items:                
            feats.append(f)
    assert len(boxes) == len(feats)

    bundles = list(zip(boxes, probs, feats))
    bundles = sorted(bundles, key=lambda x: x[0][-1], reverse = True) # sort by confidence descendingly 
    
    boxes, probs, feats = zip(*bundles)
    
    return (list(boxes), list(probs), list(feats))


def package_image_summary(im_summary, _feature_path): 
    boxes, probs, feats = formalize_bbox(im_summary)

    im_summary_out = {}
    im_summary_out['boxes'] = boxes
    im_summary_out['scale'] = im_summary.info.dim_scale[2]
    curr_im_path = im_summary.info.image_idx + ".pkl"
    pickle.dump(im_summary_out, open(os.path.join(_feature_path, curr_im_path), 'wb'))
    


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset', help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

def filter_small_box(boxes, min_area): 
  boxes_index = []
  for i, box in enumerate(boxes): 
    x1, y1, x2, y2, _ = box
    area = (x2-x1)*(y2-y1)
    if(area >= min_area): 
      boxes_index.append(i)
  return boxes_index


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


  num_classes = len(class_labels)

  image_path = os.path.join('/home/alex/faster-rcnn.pytorch/data/flickr30k_alex/') 
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
  

  args = parse_args()

  print('Called with args:')
  print(args)

  np.random.seed(cfg.RNG_SEED)
  assert args.dataset == 'vg'
  assert args.net == 'vgg16'

  args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)


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
  fasterRCNN = generic_extractor(class_labels, pretrained=False, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print('load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1).to(device)
  gt_boxes = torch.FloatTensor([[ 1.,  1.,  1.,  1.,  1.]]).to(device)
  num_boxes = torch.LongTensor([0]).to(device)

  if args.cuda:
    cfg.CUDA = True


  fasterRCNN.to(device)
  fasterRCNN.eval()


  thresh = 0.0 # default value when vis=False

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=True)
  data_iter = iter(dataloader)

  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in tqdm(range(num_images)):
      all_feat_class = [[] for _ in xrange(num_classes)]
      all_probs_class = [[] for _ in xrange(num_classes)]
      all_boxes_class = [[] for _ in xrange(num_classes)]

      data = next(data_iter)
      scale = data[1].item()

      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info = torch.FloatTensor([[im_data.size(2), im_data.size(3), scale]]).to(device)

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, image_summary = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      ###### assume: order does not change
      image_summary.info.image_idx = image_index[i]
      image_summary.info.data = generic_extractor._detach2numpy(im_data).squeeze()

      # phase 0 
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5] # (x1, y1, x2, y2)

      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data

      box_deltas = box_deltas.view(-1, 4) \
          * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
          + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
      box_deltas = box_deltas.view(1, -1, 4 * len(class_labels))

      # adjust boxes by deltas; output in (x1, y1, x2, y2)
      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      # avoid boxes go out of image
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

  
      pred_boxes /= scale # (x1, y1, x2, y2)

      scores = scores.squeeze() # torch.Size([300, 151])
      pooled_feat_backup = image_summary.pred.pooled_feat

      pred_boxes = pred_boxes.squeeze() # torch.Size([300, 604]), 604=4*151

      for j in xrange(1, num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            curr_prob = scores # 300 x 151
            curr_feat = pooled_feat_backup # 300 x 512 x 7 x 7 

            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            curr_prob = curr_prob[order]
            curr_feat = curr_feat[order]

            keep = nms(cls_dets, cfg.TEST.NMS)

            cls_dets = cls_dets[keep.view(-1).long()]
            curr_prob = curr_prob[keep.view(-1).long()]
            curr_feat = curr_feat[keep.view(-1).long()]

            all_boxes_class[j] = cls_dets.cpu().numpy()
            all_probs_class[j] = curr_prob.cpu().numpy()
            all_feat_class[j] = curr_feat
          else:
            all_boxes_class[j] = empty_array
            all_probs_class[j] = empty_array
            all_feat_class[j] = empty_array
      
      min_area = 2000
      for j in xrange(1, num_classes):
          filter_index = filter_small_box(all_boxes_class[j], min_area)
          all_boxes_class[j] = all_boxes_class[j][filter_index]
          all_probs_class[j] = all_probs_class[j][filter_index]
          all_feat_class[j] = all_feat_class[j][filter_index]

      # Limit to max_per_image detections *over all classes*
      # phase 3
      curr_boxes = []
      curr_scores = []
      if max_per_image > 0:
          # flatten scores for all boxes of this image
          image_scores = np.hstack([all_boxes_class[j][:, -1] for j in xrange(1, num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, num_classes):
                  keep = np.where(all_boxes_class[j][:, -1] >= image_thresh)[0]
                  all_boxes_class[j] = all_boxes_class[j][keep, :]
                  all_probs_class[j] = all_probs_class[j][keep, :]
                  all_feat_class[j] = all_feat_class[j][keep, :]

      if(i % 1000 == 0):
          print("Cleaning CUDA cache")
          torch.cuda.empty_cache()

      image_summary.pred.cls_prob = [all_probs_class[j] for j in range(num_classes)]
      image_summary.pred.boxes= [all_boxes_class[j] for j in range(num_classes)] 
      image_summary.pred.pooled_feat = [all_feat_class[j] for j in range(num_classes)] 

      feature_file = os.path.join(feature_path, image_summary.info.image_idx+".pkl")
      package_image_summary(image_summary, feature_path) 


